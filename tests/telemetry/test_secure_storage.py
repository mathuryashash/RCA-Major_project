"""Tests for telemetry.secure_storage: DPAPI at-rest protection."""

import pytest

from telemetry import secure_storage


pytestmark = pytest.mark.skipif(
    not secure_storage.dpapi_available(),
    reason="DPAPI requires pywin32 on Windows",
)


def test_encrypt_bytes_is_not_plaintext():
    data = b"a trained model's state_dict would go here"
    encrypted = secure_storage.encrypt_bytes(data)
    assert encrypted != data
    assert data not in encrypted


def test_encrypt_decrypt_round_trip():
    data = b"\x00\x01\x02 arbitrary binary content \xff\xfe"
    assert secure_storage.decrypt_bytes(secure_storage.encrypt_bytes(data)) == data


def test_decrypt_passes_through_plaintext_unchanged():
    """An artifact written before encryption existed must still load."""
    plaintext = b"legacy unencrypted artifact bytes"
    assert secure_storage.decrypt_bytes(plaintext) == plaintext


def test_save_and_load_encrypted_round_trip(tmp_path):
    path = tmp_path / "model.pt"
    data = b"model bytes " * 100
    secure_storage.save_encrypted(path, data)
    assert path.exists()
    on_disk = path.read_bytes()
    assert data not in on_disk, "file on disk must not contain the plaintext"
    assert secure_storage.load_encrypted(path) == data


def test_save_encrypted_is_atomic_no_leftover_tmp(tmp_path):
    path = tmp_path / "model.pt"
    secure_storage.save_encrypted(path, b"data")
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_encrypted_file_bound_to_current_user_marker(tmp_path):
    """The on-disk format carries the DPAPI magic marker, not a bare pickle."""
    path = tmp_path / "model.pt"
    secure_storage.save_encrypted(path, b"payload")
    assert path.read_bytes().startswith(secure_storage._MAGIC)


def test_decrypt_wraps_master_key_failure_in_clear_error(monkeypatch):
    """CryptUnprotectData failure (e.g. DPAPI master key gone after a
    password reset) must surface as a catchable, actionable RuntimeError,
    not a raw pywintypes.error."""
    def _boom(*args, **kwargs):
        raise OSError("simulated: DPAPI master key unavailable")

    monkeypatch.setattr(secure_storage.win32crypt, "CryptUnprotectData", _boom)
    encrypted = secure_storage._MAGIC + b"not real ciphertext but that's fine, decrypt never gets called"
    with pytest.raises(RuntimeError, match="Retrain the model"):
        secure_storage.decrypt_bytes(encrypted)


def test_encrypt_fallback_warns_once(monkeypatch, caplog):
    """When DPAPI is unavailable, the plaintext fallback must be logged,
    not silent -- otherwise a security downgrade is invisible to the user."""
    monkeypatch.setattr(secure_storage, "DPAPI_AVAILABLE", False)
    monkeypatch.setattr(secure_storage, "_warned_plaintext_fallback", False)
    with caplog.at_level("WARNING"):
        secure_storage.encrypt_bytes(b"data")
    assert any("unencrypted" in record.message for record in caplog.records)
