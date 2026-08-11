# Documentation

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system is put together, module layout, data model, threading, packaging, and where the design is weakest |
| [DECISIONS.md](DECISIONS.md) | Every library choice, what it was chosen over, and what it has cost |
| [MODEL_RATIONALE.md](MODEL_RATIONALE.md) | Why an LSTM autoencoder, why Granger causality, alternatives rejected, hyperparameters, and limits |
| [WORKFLOW.md](WORKFLOW.md) | End to end, install to report, with measured timings |
| [IMPLEMENTATION_PAPER.md](IMPLEMENTATION_PAPER.md) | The formal write-up, built from instrumented runs |
| [INTERVIEW_PREP.md](INTERVIEW_PREP.md) | Questions to expect, with grounded answers |
| [RESUME_MATERIAL.md](RESUME_MATERIAL.md) | CV bullets with the measurements behind them, and the claims to avoid |
| [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) | What stands between this and a product you would hand to a stranger, with verification checks |

Setup and usage live outside this folder: [INSTALL.md](../INSTALL.md) and the
[README](../README.md).

---

## If you read one thing

[MODEL_RATIONALE.md](MODEL_RATIONALE.md) §5–6 and
[IMPLEMENTATION_PAPER.md](IMPLEMENTATION_PAPER.md) §5. They cover the result
that shaped most of the later work: the causal layer produces no supported
chain on the majority of real incidents, the ranking degenerates when that
happens, and the useful response was to make the system say so rather than to
manufacture an answer.

## A note on these documents

They are written from measured runs on a live installation, so they record
failures alongside features — 27.8% collection coverage, packaging bugs that
only appeared in the frozen build, fixes that turned out worse than the
defects they addressed. That is deliberate. Anything stated as measured was
measured; anything unestablished says so.
