"""
LogBERT-based log anomaly detector.

Uses a DistilBERT model fine-tuned with Masked Language Modeling (MLM) on
a healthy log corpus. At inference, tokens are masked and prediction loss
is used as the anomaly signal — logs the model cannot predict well are
likely anomalous.

Requires: transformers, torch (CPU-only is fine).
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import flag — lets the rest of the codebase check availability
# without triggering an ImportError at module load time.
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import (
        DistilBertForMaskedLM,
        DistilBertTokenizerFast,
        DataCollatorForLanguageModeling,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LogBERTDetector:
    """Anomaly detector that fine-tunes DistilBERT via MLM on healthy logs.

    Scoring strategy:
    1. Tokenize a log message.
    2. Randomly mask 15 % of sub-word tokens (same scheme as BERT pre-training).
    3. Compute the model's cross-entropy loss on the masked positions.
    4. Aggregate per-token loss into a single score per message.
    5. Normalise to [0, 1] using a sigmoid centred on the 95th-percentile
       loss observed during training.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_seq_length: int = 128,
        device: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "LogBERTDetector requires `transformers` and `torch`. "
                "Install them with: pip install transformers torch"
            )

        self.model_name = model_name
        self.max_seq_length = max_seq_length

        # Device selection — default to CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained tokenizer + model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)

        # Training bookkeeping
        self._trained = False
        self._threshold: float = 1.0  # 95-th percentile raw loss; set during train()
        self._loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether the model has been fine-tuned on a log corpus."""
        return self._trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        log_messages: List[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict:
        """Fine-tune the model on *healthy* logs using MLM.

        Parameters
        ----------
        log_messages : list[str]
            Corpus of healthy (non-anomalous) log messages.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.  If you're on CPU and see OOM-like slowness,
            lower this.
        learning_rate : float
            AdamW learning rate.

        Returns
        -------
        dict
            ``{"loss_history": [...], "final_loss": float, "epochs": int}``
        """
        if not log_messages:
            raise ValueError("log_messages must be a non-empty list of strings.")

        # Filter out empty strings
        log_messages = [m for m in log_messages if m.strip()]
        if not log_messages:
            raise ValueError("All log_messages were empty strings.")

        # Tokenize the full corpus
        encodings = self.tokenizer(
            log_messages,
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Build a simple map-style dataset
        dataset = _EncodingDataset(encodings)

        # MLM data collator — masks 15 % of tokens each batch
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        # DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        # Optimiser
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        self._loss_history = []

        for epoch in range(epochs):
            epoch_losses: List[float] = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            self._loss_history.append(mean_loss)
            logger.info(
                "Epoch %d/%d — mean MLM loss: %.4f", epoch + 1, epochs, mean_loss
            )

        # Compute per-message raw scores on the training corpus to establish
        # the normalisation threshold (95th percentile).
        self.model.eval()
        raw_scores = self._raw_scores(log_messages, batch_size=batch_size)
        self._threshold = float(np.percentile(raw_scores, 95))
        # Guard against degenerate case where threshold is 0 or negative
        if self._threshold <= 0:
            self._threshold = 1.0

        self._trained = True

        return {
            "loss_history": list(self._loss_history),
            "final_loss": self._loss_history[-1],
            "epochs": epochs,
            "threshold_95": self._threshold,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, log_messages: List[str], batch_size: int = 32) -> np.ndarray:
        """Return anomaly scores in [0, 1] for each log message.

        Higher values ⇒ more anomalous (model predicts tokens poorly).

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self._trained:
            raise ValueError("Model has not been fine-tuned yet. Call train() first.")

        if not log_messages:
            return np.array([], dtype=np.float64)

        # Track which inputs are empty so we can force their score to 0
        empty_mask = np.array(
            [not m or not m.strip() for m in log_messages], dtype=bool
        )

        raw = self._raw_scores(log_messages, batch_size=batch_size)
        normalised = self._normalise(raw)

        # Empty / whitespace-only inputs should always score 0.0
        normalised[empty_mask] = 0.0
        return normalised

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fine-tuned model, tokenizer and metadata."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))

        # Save detector metadata
        meta = {
            "threshold": self._threshold,
            "trained": self._trained,
            "max_seq_length": self.max_seq_length,
            "loss_history": self._loss_history,
        }
        np.save(str(out / "logbert_meta.npy"), meta)
        logger.info("LogBERTDetector saved to %s", out)

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Model directory not found: {src}")

        self.model = DistilBertForMaskedLM.from_pretrained(str(src))
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(src))
        self.model.to(self.device)

        meta_path = src / "logbert_meta.npy"
        if meta_path.exists():
            meta = np.load(str(meta_path), allow_pickle=True).item()
            self._threshold = float(meta.get("threshold", 1.0))
            self._trained = bool(meta.get("trained", False))
            self.max_seq_length = int(meta.get("max_seq_length", 128))
            self._loss_history = list(meta.get("loss_history", []))
        else:
            # Backwards compat — assume it was trained if we loaded weights
            self._trained = True
            self._threshold = 1.0

        logger.info("LogBERTDetector loaded from %s (trained=%s)", src, self._trained)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_scores(self, log_messages: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute raw mean-cross-entropy per message.

        Scoring masks **all** non-special tokens and computes the mean
        cross-entropy loss over them.  This is fully deterministic (no
        randomness) and maximally discriminative — the model must predict
        every content token from context alone.
        """
        self.model.eval()

        # Handle edge cases: replace empty / whitespace-only strings with a
        # placeholder so the tokenizer doesn't produce degenerate tensors.
        cleaned: List[str] = []
        empty_indices: set = set()
        for i, msg in enumerate(log_messages):
            if not msg or not msg.strip():
                cleaned.append("[UNK]")
                empty_indices.add(i)
            else:
                cleaned.append(msg)

        scores: List[float] = []
        for start in range(0, len(cleaned), batch_size):
            batch_texts = cleaned[start : start + batch_size]

            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            input_ids: torch.Tensor = encodings["input_ids"]  # (B, L)
            attention_mask: torch.Tensor = encodings["attention_mask"]

            # Mask ALL non-special, non-padding tokens (deterministic)
            special_mask = self._special_token_mask(input_ids)
            mask_positions = (~special_mask) & (attention_mask.bool())

            labels = input_ids.clone()
            labels[~mask_positions] = -100  # ignore special / padding tokens

            mask_token_id = self.tokenizer.mask_token_id
            masked_input = input_ids.clone()
            masked_input[mask_positions] = mask_token_id

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=masked_input.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    labels=labels.to(self.device),
                )

            # Per-token loss for per-sample aggregation
            logits = outputs.logits  # (B, L, V)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.to(self.device).view(-1),
            ).view(input_ids.shape)  # (B, L)

            # Mean loss over masked positions per sample
            for i in range(per_token_loss.size(0)):
                sample_mask = mask_positions[i]
                if sample_mask.sum() == 0:
                    scores.append(0.0)
                else:
                    scores.append(per_token_loss[i][sample_mask].mean().item())

        result = np.array(scores, dtype=np.float64)

        # Override empty-input positions with 0.0
        for idx in empty_indices:
            if idx < len(result):
                result[idx] = 0.0

        return result

    def _normalise(self, raw_scores: np.ndarray) -> np.ndarray:
        """Sigmoid normalisation centred on the training threshold."""
        # sigmoid( (x - threshold) / scale )
        # scale controls steepness; we use threshold / 4 so that the
        # transition band is roughly ±2·scale around the threshold.
        scale = max(self._threshold / 4.0, 1e-6)
        normalised = 1.0 / (1.0 + np.exp(-(raw_scores - self._threshold) / scale))
        return np.clip(normalised, 0.0, 1.0)

    def _special_token_mask(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Return a boolean tensor that is True for special tokens."""
        special_ids = set(self.tokenizer.all_special_ids)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            mask |= input_ids == sid
        return mask


# ---------------------------------------------------------------------------
# Simple dataset wrapper
# ---------------------------------------------------------------------------


class _EncodingDataset(torch.utils.data.Dataset):
    """Wraps a BatchEncoding dict as a map-style Dataset."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
