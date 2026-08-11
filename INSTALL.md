# Installing LocalRCA

Windows 10 or 11, 64-bit. No administrator rights are required, and nothing
is written outside your own user profile.

---

## 1. Get the release

Download `LocalRCA-vX.Y.Z-windows-x64.zip` from the
[Releases page](https://github.com/mathuryashash/RCA-Major_project/releases)
and extract **the whole ZIP** to a folder you can write to, for example
`C:\LocalRCA`.

You should end up with:

```
C:\LocalRCA\
├── RCA-Desktop\
│   ├── RCA-Desktop.exe        ← the application
│   └── _internal\             ← runtime; must stay beside the .exe
└── RCA-Collector\
    ├── RCA-Collector.exe      ← the background collector
    └── _internal\
```

Do not move either `.exe` out of its folder. Move the whole folder if you
need to relocate it.

Windows may show a SmartScreen prompt: the build is unsigned. Check the
release is published from this repository before choosing **Run anyway**.

## 2. Start the application

Run `RCA-Desktop\RCA-Desktop.exe`.

On first launch it shows exactly what will be recorded — system metrics every
30 seconds, the busiest process names every 5 minutes, and an allowlist of
Windows Event Log entries — and asks whether to begin. **Nothing is collected
until you agree.**

## 3. Keep collecting when the app is closed

Optional, but the tool is far more useful with it: the collector needs a
continuous baseline, and it can only build one while it is running.

Open PowerShell in the extracted folder:

```powershell
.\RCA-Collector\RCA-Collector.exe install
.\RCA-Collector\RCA-Collector.exe status
```

`install` registers a per-user startup entry, adds a Start menu shortcut, and
lists the app in Add/Remove Programs. `status` reports consent, whether the
collector is scheduled, and how much telemetry exists so far.

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
| `install` | Startup entry, Start menu shortcut, Add/Remove Programs entry |
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
