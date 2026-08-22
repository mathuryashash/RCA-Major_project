# Installing LocalRCA

Windows 10 or 11, 64-bit. No administrator rights are required, and nothing
is written outside your own user profile.

---

## What it will cost you

Decide before downloading, not afterwards:

| | |
|---|---|
| Download | **272 MB** (ZIP) |
| After extraction | **731 MB** |
| Collected data | **~3.3 MB per day**, capped by retention |
| Memory while open | ~430 MB |
| Before it is useful | **~21 hours** of collection |

Most of what remains is PyTorch, which runs the model: 351 MB of the 731. The
figures used to add roughly another 360 MB of bundled web browser, which
version 1.5.0 removed by drawing them natively instead.

The data footprint is bounded — metrics expire after 365 days, process detail
and the foreground-application record after 30 — and freed space is returned
to the filesystem rather than left inside the database file.

## About the SmartScreen warning

**This build is not code-signed, and Windows will say so.** You will see
*"Windows protected your PC"* with an *Unknown publisher*.

That warning is Windows telling you the truth: nobody has paid a certificate
authority to vouch for this binary. A code-signing certificate costs a few
hundred dollars a year, and this project does not have one.

If you choose to proceed, verify what you downloaded first — that is what the
checksum in step 1 is for. It confirms the file is byte-for-byte what was
published, which is the assurance signing would otherwise give you. If you are
not comfortable with that, building from source is a reasonable alternative and
is documented in the README.

---

## 1. Get the release

Download `LocalRCA-vX.Y.Z-windows-x64.zip` from the
[Releases page](https://github.com/mathuryashash/RCA-Major_project/releases).

**Verify it before running it:**

```powershell
Get-FileHash LocalRCA-vX.Y.Z-windows-x64.zip -Algorithm SHA256
```

Compare the result with the SHA256 published beside the download. If they
differ, stop — the file is not what was released.

Then extract **the whole ZIP** to a folder you can write to, such as
`C:\LocalRCA`. Right-click the ZIP and choose **Extract All**; opening the ZIP
and dragging one file out will not work, because the application needs the
runtime folder beside it.

You should end up with:

```
C:\LocalRCA├── Install LocalRCA.bat       ← start here
├── RCA-Desktop│   ├── RCA-Desktop.exe        ← the application
│   └── _internal\             ← runtime; must stay beside the .exe
└── RCA-Collector    ├── RCA-Collector.exe      ← the background collector
    └── _internal```

Do not move either `.exe` out of its folder. Move the whole folder if you need
to relocate it.

## 2. Start the application

**Double-click `Install LocalRCA.bat`.** It checks the ZIP was extracted
properly rather than opened, warns if the drive is short of space, explains
what agreeing will set up, and then starts the application.

If you would rather not run a batch file, run
`RCA-Desktop\RCA-Desktop.exe` directly — it does the same thing without the
checks.

> **The first launch is slow — around a minute — and that is expected.**
> Windows Defender scans several thousand freshly extracted files the first
> time they are touched. Measured here: **52 seconds** on the first run and
> **9.5 seconds** on every run after. Nothing is wrong; give it a minute.

On first launch it shows exactly what will be recorded — system metrics every
30 seconds, the busiest process names every 5 minutes, and an allowlist of
Windows Event Log entries — and asks whether to begin. **Nothing is collected
until you agree.**

## 3. That is the installation

Agreeing in step 2 finishes the setup: the collector is registered to start at
every logon, the app is added to your Start menu, and **LocalRCA** appears in
Add / Remove Programs. No administrator rights, nothing outside your user
profile, and no command line.

To check on it at any point:

```powershell
.\RCA-Collector\RCA-Collector.exe status
```

## 4. Wait for a baseline

The model learns *your* machine's normal behaviour, so there is nothing to
train against until enough clean telemetry exists — roughly **21 hours of
uninterrupted collection**. Stage 1 shows the remaining time and unlocks the
Train button by itself.

Training then takes under a minute. Stage 2 unlocks once a model exists.

---

## Commands

Run these from the extracted folder as `.\RCA-Collector\RCA-Collector.exe <command>`.

| Command | What it does |
|---|---|
| `accept-consent` | Grants consent from the command line instead of the dialog |
| `install` | Registers all three by hand. The app does this on first launch, so it is only needed if you declined and changed your mind |
| `status` | Consent, schedule state, sample count |
| `uninstall` | Stops collection and removes all three. **Keeps your data** |
| `delete-all-data` | Erases everything collected. See below |
| `run` | Runs the collector in the foreground |

### Uninstalling

Either remove **LocalRCA** from Windows Settings → Apps → Installed apps, or:

```powershell
.\RCA-Collector\RCA-Collector.exe uninstall
```

Both stop collection and remove the startup entry, the Start menu shortcut
and the Add/Remove Programs entry. Your collected data is **deliberately
kept** — removing autostart should not silently discard a trained model.

To erase the data as well:

```powershell
.\RCA-Collector\RCA-Collector.exe delete-all-data
```

That removes the whole of `%LOCALAPPDATA%\RCA` — the database, the trained
model, every generated report and the logs — plus any charts left in your
temp directory. Reports you exported yourself are not touched, because you
chose where those went.

Then delete the extracted folder.

---

## Running from source

```powershell
git clone https://github.com/mathuryashash/RCA-Major_project.git
cd RCA-Major_project
pip install -r requirements.txt

$env:PYTHONPATH = "$PWD\src"
python -m desktop.main
```

The CLI is `python -m telemetry <command>` with the same commands as above.
A source checkout deliberately does not create a Start menu shortcut or an
Add/Remove Programs entry — a checkout should not present itself as installed
software.

To build the executables yourself:

```powershell
.\packaging\build.ps1        # ~19 minutes, produces dist\RCA-Desktop and dist\RCA-Collector
```

---

## Where things live

| Path | Contents |
|---|---|
| `%LOCALAPPDATA%\RCA\telemetry.db` | Collected telemetry |
| `%LOCALAPPDATA%\RCA\telemetry_model.pt` | The trained model |
| `%LOCALAPPDATA%\RCA\reports\` | Generated reports |
| `%LOCALAPPDATA%\RCA\collector.log` | Collector log |
| `%LOCALAPPDATA%\RCA\desktop.log` | Desktop app log — **send this with a bug report** |
| `%APPDATA%\...\Startup\rca-collector.cmd` | The logon entry |

## If something goes wrong

`desktop.log` records unhandled errors and names the version that produced
them. It is the first thing to check, and the most useful thing to attach to
an issue.

Nothing in this application makes a network connection. If collection seems
to have stopped, run `status` — the collector does not survive a crash on its
own and restarts at the next logon, or when you next open the desktop app.
