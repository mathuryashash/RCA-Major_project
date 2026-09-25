LocalRCA v1.5.2 — smaller, harder to stop by accident, easier to report bugs

Download LocalRCA-v1.5.2-windows-x64.zip, extract it, and run "Install LocalRCA.bat".

## What changed

**7x smaller download.** The app now ships the CPU build of PyTorch instead of the CUDA build, which had been pulling in 2.6 GB of NVIDIA libraries the app never used.

| | v1.5.1 | v1.5.2 |
|---|---|---|
| Download | 1.99 GB | 268 MB |
| Installed | 3.0 GB | 720 MB |

Every dependency is now pinned to an exact version and hash, and GitHub runs the full test suite and lint on every push.

**Collection no longer stops silently.**
- If nothing has been recorded for more than 5 minutes, the app shows a red warning with a "Restart collection" button.
- The background supervisor used to exit permanently if the collector program went missing (for example during an update). It now waits up to an hour for it to come back.
- On a PC shared by several Windows accounts, one user's collector no longer blocks another user's.
- The supervisor now works on accounts whose folder name has accented or non-English characters (for example `C:\Users\José`). Previously it could never find the collector there.
- Pausing collection now stays paused. Opening the app, or signing in again, used to switch collection back on while the app still showed "Paused".

**Export Diagnostics.** A new button in the header saves a zip of logs, version and health information to attach to a bug report. It never includes the telemetry database, the trained model or any collected data, and it removes your username and home folder path. A GitHub bug-report template asks for this file.

## Verification

- 195 automated tests pass; lint clean; CI green.
- Both executables are signed with a timestamped self-signed certificate. This proves the files haven't been changed since the build, but it does not remove the Windows SmartScreen warning ("More info" → "Run anyway").
- SHA256 (`LocalRCA-v1.5.2-windows-x64.zip`):
  `440085f2e0ee960593a5963613c820fbbbe6b10b4118fc157e54af47a8db50cd`

## Known limitations

- The telemetry database is not encrypted on Windows 11 Home, which has no EFS.
- For about a minute after waking from sleep, the app may briefly show the "collection stopped" warning.
- If the collector program is missing for more than an hour, the supervisor gives up until the next sign-in.
