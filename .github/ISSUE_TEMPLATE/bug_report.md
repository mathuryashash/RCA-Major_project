---
name: Bug report
about: Something in LocalRCA is broken, wrong, or stopped collecting
title: "[Bug] "
labels: bug
assignees: ''
---

## What happened

<!-- A clear description of the problem. What did you see? -->

## What you expected

<!-- What should have happened instead? -->

## Steps to reproduce

1.
2.
3.

## Diagnostics bundle (please attach)

LocalRCA never sends anything off your machine by itself, so we can't see
what went wrong unless you share it. In the app, click **Export
diagnostics…** (top-right, next to *Advanced*), save the zip, and drag it
into this issue.

**What the zip contains**

- `system.json`: LocalRCA version, Windows version/edition, Python version
- `health.json`: collection health as *counts only*: number of samples and
  events, coverage %, sampling gaps, first/last sample time, database size,
  whether the collector is registered or paused, whether a model file exists
- `logs/`: the collector and desktop logs (with rotated copies) and the
  collector launcher scripts

**What it does NOT contain**

- no telemetry rows: none of your collected samples, events, process lists
  or foreground-app history
- no database file, no trained model, no generated RCA reports

Your Windows username, profile path and computer name are replaced with
`<user>` / `<computer>` in every file. Log lines can still name a program
that appeared in an error message, so if that matters to you, open the zip
and read `logs/` before attaching it (it's plain text).

- [ ] I attached the diagnostics zip
- [ ] I couldn't export it (tell us why below, and paste the version shown in
      the window title)

## Anything else

<!-- Screenshots, when it started, what changed recently (Windows update,
new hardware, sleep/hibernate), etc. -->
