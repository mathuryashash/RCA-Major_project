"""Best-effort redaction for opted-in Event Log values."""

import re

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_USER_DIR = re.compile(r"\b([A-Za-z]):\\Users\\[^\\\s\"']+", re.IGNORECASE)
_UNC = re.compile(r"\\\\[^\\\s\"']+\\[^\\\s\"']+")


def redact(text: str | None, username: str | None = None, max_len: int = 512) -> str:
    if not text:
        return ""
    output = _URL.sub("<url redacted>", text)
    output = _EMAIL.sub("<email redacted>", output)
    output = _USER_DIR.sub(lambda match: f"{match.group(1)}:\\Users\\<redacted>", output)
    output = _UNC.sub(r"\\\\<redacted>", output)
    if username:
        output = re.sub(re.escape(username), "<redacted>", output, flags=re.IGNORECASE)
    return output[:max_len]
