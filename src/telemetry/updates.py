"""Opt-in check for a newer release.

This is the one part of the application that can touch the network, and it
exists because the alternative is worse: without it, a defect that ships is a
defect that stays on every machine that has it. This project has found nine
defects in code carrying a passing test suite, two of which were fixes worse
than the bugs they repaired, so "we will simply not ship bugs" is not a claim
available here.

It is deliberately the smallest thing that solves that problem:

* **Off unless switched on.** No check happens until the user asks for one.
* **Reads one number.** It fetches the newest release tag and compares it to
  the running version. Nothing is downloaded, nothing is installed, and the
  user is told rather than acted upon.
* **Sends nothing.** A plain GET to a public endpoint. No identifier, no
  telemetry, no query string. The request body is empty and stays empty --
  if a future change needs to *send* something, that is a different feature
  and needs its own consent.
* **Fails quietly.** Offline, blocked, rate-limited or DNS-poisoned all look
  the same to a user who did not ask for a network: nothing happens.

The privacy claim elsewhere in this project has to be stated precisely as a
result. The collector never opens a socket under any configuration. The
desktop application opens one only when a user has turned this on and only
while the check runs.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import store
from .logsetup import get_logger

_LOGGER = get_logger(__name__)

#: Public, unauthenticated, and returns the newest published release. Draft and
#: pre-release entries are excluded by GitHub itself, so an unpublished draft
#: is invisible here -- which is what allows a release to be staged without
#: every installed copy announcing it.
RELEASES_API = (
    "https://api.github.com/repos/mathuryashash/RCA-Major_project/releases/latest"
)
RELEASES_PAGE = "https://github.com/mathuryashash/RCA-Major_project/releases"

#: Short. A user who asked for a check wants an answer or nothing, not a
#: window that hangs while a captive portal decides.
TIMEOUT_SECONDS = 6

_META_KEY = "update_check_enabled"


@dataclass(frozen=True)
class UpdateStatus:
    """What a check found. `available` is only ever True on a real comparison."""

    checked: bool
    available: bool = False
    latest: str | None = None
    current: str | None = None
    url: str = RELEASES_PAGE
    reason: str = ""


def is_enabled(conn) -> bool:
    return store.get_meta(conn, _META_KEY, "0") == "1"


def set_enabled(conn, enabled: bool) -> None:
    store.set_meta(conn, _META_KEY, "1" if enabled else "0")


def _parse(version: str) -> tuple[int, ...]:
    """`v1.4.1` -> (1, 4, 1). Unparseable input sorts lowest.

    Compared as integers, not as text: "1.10.0" is newer than "1.9.0" and a
    string comparison says the opposite, which would tell everyone running the
    newer build to downgrade.
    """
    found = re.findall(r"\d+", version or "")
    return tuple(int(part) for part in found[:3]) if found else (0,)


def check(current_version: str, *, conn=None, force: bool = False) -> UpdateStatus:
    """Ask whether a newer release exists.

    Returns rather than raises: every failure mode here -- offline, blocked,
    rate-limited, malformed response -- is a normal state for a desktop
    application and none of them is worth interrupting the user over.
    """
    if conn is not None and not force and not is_enabled(conn):
        return UpdateStatus(checked=False, reason="update checks are turned off")

    request = urllib.request.Request(
        RELEASES_API,
        headers={
            # GitHub requires a User-Agent. It names the software and nothing
            # about the machine or the person running it.
            "User-Agent": f"LocalRCA/{current_version}",
            "Accept": "application/vnd.github+json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return UpdateStatus(checked=False, reason=f"could not reach the release page: {exc}")
    except (ValueError, json.JSONDecodeError) as exc:
        return UpdateStatus(checked=False, reason=f"unexpected response: {exc}")

    latest = (payload or {}).get("tag_name")
    if not latest:
        # No published release yet -- which is the normal state while a release
        # is still a draft, and not an error.
        return UpdateStatus(checked=True, reason="no published release found")

    newer = _parse(latest) > _parse(current_version)
    if newer:
        _LOGGER.info("A newer release is available: %s (running %s)", latest, current_version)
    return UpdateStatus(
        checked=True,
        available=newer,
        latest=latest,
        current=current_version,
        url=(payload.get("html_url") or RELEASES_PAGE),
        reason="" if newer else "running the newest release",
    )
