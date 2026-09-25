"""At-rest protection for files this application writes to disk.

Two independent Windows-native mechanisms, layered rather than chosen
between, since either can be unavailable (DPAPI needs pywin32; EFS needs an
NTFS volume and is absent on Windows Home for some editions):

- DPAPI (``CryptProtectData``) wraps the trained model artifact's bytes so
  they are unreadable without the same Windows user's login secrets --
  copying ``telemetry_model.pt`` to another machine, or reading it as another
  user on this one, yields ciphertext. This is the same primitive Windows
  Credential Manager and saved browser passwords use; it needs no
  certificate, no key management, and nothing for the user to configure.
- EFS (``cipher.exe /e``) is applied to the whole application data folder,
  covering telemetry.db and the log files that DPAPI does not wrap. It is
  best-effort: unsupported editions or non-NTFS volumes report failure,
  which is logged, not raised -- collection must not fail to start because
  encryption could not be enabled.

Neither mechanism protects against an attacker with an unlocked, logged-in
session as the same user -- that boundary is Windows login itself, which
this application does not attempt to strengthen.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .logsetup import get_logger

_LOGGER = get_logger(__name__)

try:
    import win32crypt
    DPAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only off Windows
    win32crypt = None
    DPAPI_AVAILABLE = False

#: Prefixes an encrypted artifact so an already-deployed plaintext file
#: (written before this module existed) is still readable: decrypt_bytes
#: passes anything without this marker straight through.
_MAGIC = b"RCADPAPI1"


def dpapi_available() -> bool:
    return DPAPI_AVAILABLE


def encrypt_bytes(data: bytes, description: str = "LocalRCA model artifact") -> bytes:
    """Wrap ``data`` with DPAPI bound to the current Windows user.

    Falls back to returning ``data`` unchanged when DPAPI is not available
    (non-Windows, or pywin32 missing) rather than raising -- a model that
    cannot be encrypted must still be usable, since refusing to save it is a
    worse outcome for a single-user local tool than an unencrypted file. The
    fallback is logged once per process so the downgrade is visible in
    collector.log/desktop.log rather than only in this docstring.
    """
    if not DPAPI_AVAILABLE:
        _warn_plaintext_fallback_once()
        return data
    encrypted = win32crypt.CryptProtectData(data, description, None, None, None, 0)
    return _MAGIC + encrypted


_warned_plaintext_fallback = False


def _warn_plaintext_fallback_once() -> None:
    global _warned_plaintext_fallback
    if not _warned_plaintext_fallback:
        _warned_plaintext_fallback = True
        _LOGGER.warning(
            "DPAPI is unavailable on this machine (non-Windows, or pywin32 "
            "missing); the trained model artifact will be saved unencrypted. "
            "Every subsequent save this process makes is also plaintext."
        )


def decrypt_bytes(data: bytes) -> bytes:
    """Reverse :func:`encrypt_bytes`, transparently passing through plaintext."""
    if not data.startswith(_MAGIC):
        return data
    if not DPAPI_AVAILABLE:
        raise RuntimeError(
            "This artifact was DPAPI-encrypted but pywin32 is unavailable to "
            "decrypt it on this machine."
        )
    payload = data[len(_MAGIC):]
    try:
        _description, decrypted = win32crypt.CryptUnprotectData(payload, None, None, None, 0)
    except Exception as exc:
        # CryptUnprotectData fails with a raw pywintypes.error when the
        # Windows account's DPAPI master key is gone or unrecoverable (most
        # commonly: the account password was reset without the old password
        # or a recovery agent). Re-raised as a clear, catchable error instead
        # of a native OS exception type leaking into callers -- the correct
        # user-facing action is "retrain", not a cryptic Windows error string.
        raise RuntimeError(
            "This artifact could not be decrypted -- Windows account "
            "credentials likely changed since it was saved. Retrain the "
            "model."
        ) from exc
    return decrypted


def save_encrypted(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` as DPAPI ciphertext, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encrypt_bytes(data))
    tmp.replace(path)


def load_encrypted(path: str | Path) -> bytes:
    """Read and decrypt bytes written by :func:`save_encrypted`."""
    return decrypt_bytes(Path(path).read_bytes())


def enable_folder_encryption(folder: str | Path, timeout_s: float = 30.0) -> bool:
    """Best-effort: turn on Windows EFS for every file in ``folder``.

    Returns whether it succeeded. Never raises -- called once at collector
    startup, and a machine without EFS support (some Home editions, non-NTFS
    volumes, already-encrypted parent folder) must still be able to collect;
    losing the feature is acceptable, losing the application is not.
    """
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["cipher.exe", "/e", "/s:" + str(folder)],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if result.returncode == 0:
            return True
        _LOGGER.warning(
            "EFS encryption of %s reported failure (code %s): %s",
            folder, result.returncode, (result.stdout or result.stderr or "").strip()[:300],
        )
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        _LOGGER.warning("EFS encryption of %s not available: %s", folder, exc)
        return False
