LocalRCA records what your computer is doing, learns what normal looks like for
*your* machine, and afterwards tries to explain a slowdown, stall or crash. It
runs entirely on the endpoint. Nothing collected is ever uploaded, and the
collector opens no sockets. The only network capability is an opt-in check for
a newer release, off until you turn it on, which reads a version number and
sends nothing.

## Before you download

**Windows will warn you, and it is right to.** This build is not code-signed,
so SmartScreen shows *"Windows protected your PC"*. Click **More info → Run
anyway** if you trust the source. If you would rather not, that is a reasonable
call — verify the checksum below first, or build from source.

**It needs about a day before it is useful.** Roughly 21 hours of clean
collection are required before there is enough baseline to train a model.
Until then the app runs and shows what it has captured, but cannot analyse
anything. This is inherent: "normal" has to be learned from your machine,
because there is no such thing as a generic normal.

## Install

1. Download `LocalRCA-v1.5.0-windows-x64.zip` (272 MB).
2. **Verify it** — in PowerShell:
   ```powershell
   Get-FileHash LocalRCA-v1.5.0-windows-x64.zip -Algorithm SHA256
   ```
   Expected:
   ```
   24D1D091DDB60A27D5EA79F0E90A5CDECABD7F213DCA3AEC459BA8B386720B65
   ```
3. Extract the **whole** ZIP somewhere you can write to, e.g. `C:\LocalRCA`.
   Do not run it from inside the archive — it needs the files beside it.
4. Run `RCA-Desktop\RCA-Desktop.exe`.
5. Read the consent dialog and agree, or decline and nothing is collected.

No administrator rights are used and nothing is written outside your own user
profile.

## What it records

Every 30 seconds: CPU, memory, disk, network, battery and GPU readings, the
name of the application currently in focus, and how long since you touched the
keyboard. Every 5 minutes — and every 30 seconds while the machine is busy —
the names of the busiest programs. As they occur: a fixed list of Windows Event
Log entries such as crashes and unexpected shutdowns.

**Never recorded:** window titles, page addresses, keystrokes, file contents,
browsing history, or the text of documents.

The application name and idle timer together form a record of when the machine
was in use and roughly what for. That is the most personal thing collected, so
it is the shortest-lived — erased after 30 days, while the numeric readings are
kept for a year. Everything is stored at `%LOCALAPPDATA%\RCA`.

**You can stop it.** *Captured Data → Pause collection* halts recording within
about 30 seconds and it stays stopped across restarts until you resume.

## Removing it

Add / Remove Programs, or `RCA-Collector.exe uninstall`, reverses the logon
entry, Start menu shortcut and Add/Remove entry, and stops collection while
keeping what was already collected. `RCA-Collector.exe delete-all-data` erases
the database, model, reports and logs as well.

## What is new in 1.5.0

- **A third smaller.** The download falls from 433 MB to **272 MB** and the
  installed size from 1.1 GB to **731 MB**. The figures used to be drawn by
  Plotly, which renders HTML and therefore needed a bundled web browser —
  about 360 MB of QtWebEngine to draw two charts. They are now drawn into a
  native Qt canvas instead. Pan and zoom still work; hover tooltips are the
  one thing lost. Rendered figures also no longer pass through temporary HTML
  files, which had been leaving metric values in your temp directory.
- **Simpler install.** `Install LocalRCA.bat` in the archive root checks the
  ZIP was extracted rather than opened — the way people actually fail at this —
  warns if the drive is short of space, explains what agreeing sets up, and
  starts the application. Running the `.exe` directly still works.
- **An update check**, off until you turn it on. It reads the newest release
  version and nothing else: no telemetry, no identifier, no download. It is the
  only part of the application that touches the network, and it never runs on
  its own.

- **Pause and resume collection** from the interface. Previously the only way
  to stop was to uninstall, which is not a defensible position for a tool whose
  case rests on privacy. Resuming restores supervision, not just the collector.
- **The window renders at the minimum size it advertises.** It declared
  1024×640 and could not honour it — content was clipped with no way to reach
  it, which also affected anyone running at 150% display scaling. Every tab now
  scrolls; the window minimum fell from 1155×180 to 532×180.
- **No stray scrollbars on a normal display.** Making every tab scrollable
  introduced a second, page-level bar beside the ones the tables and figures
  already carry. Figures now ask for a modest height and expand to fill, so the
  page fits when there is room and scrolls only when there is not.
- **The header renders on one line**, rather than as a narrow vertical column
  of single words — a regression introduced by the resizing work above and
  fixed before release.
- **Structural contrast meets WCAG SC 1.5.01.** Text was always legible at
  14.64:1, but the frames, input outlines and table borders around it rendered
  at 1.32:1 and are now 3.07:1.
- Keyboard focus is now visible on every focusable control, including the
  Event Log opt-in checkbox and the training sliders.

## Known limitations

These are stated plainly because you are about to run this on your own machine.

- **Unsigned.** Every user meets a SmartScreen warning. Mitigated only by the
  published checksum above.
- **Updates are opt-in and manual.** *Captured Data → Check for updates* asks
  once, then reads the newest release tag and tells you. It downloads and
  installs nothing; upgrading means fetching the new ZIP yourself.
- **No crash reporting.** `desktop.log` never leaves your machine — which is
  the promise working as intended, and also means problems are invisible to the
  developer unless you send that file.
- **Tested on one machine.** Every measurement in the documentation comes from
  a single Windows 11 host.
- **It often declines to explain.** Measured across 175 real incidents, roughly
  one in six produced a supported causal chain. Nearly half were too short to
  test at all. The application says so rather than inventing an answer, which
  is deliberate — but it does mean a confident explanation is the exception.

`docs/IMPLEMENTATION_PAPER.md` in the repository documents how it works, what
was measured, and where the measurements contradicted the design.
