# Production Readiness Assessment

Assessed against commit `97d4604` on 2026-09-25. Re-verify before acting on
anything here; every number below was measured on that commit, on the
development machine, and goes stale as the code moves.

This supersedes `PRODUCTION_CHECKLIST.md` (last updated 2026-08-21). That file
still lists code signing, the update check and settings persistence as missing;
all three have since shipped in some form.

"Production grade" here means: a stranger can download it, install it, leave
it running for months, and trust what it tells them, without the author on
hand. It does not mean enterprise fleet software. Each area gives the minimum
bar for a one-to-four-person project and names what would be overkill.

---

## Scorecard

| Area | State | Minimum bar met? |
|---|---|---|
| Core value (does it work?) | Explains 17.8% of real incidents; validated on one machine | No |
| Collection reliability | 85.5% coverage over 30 days; silent stop after rebuild/move | No |
| Security and privacy | Strong design; database plaintext on Windows Home | Partly |
| Packaging and install | ZIP plus a `.bat`, self-signed, 1.99 GB download | No |
| Updates | Opt-in "is there a newer version" check only | Partly |
| Supportability | Local logs only, no diagnostics export | No |
| Build and release engineering | Manual, one machine, no CI, unpinned dependencies | No |
| Testing | 150 tests, all pass, 2 min 52 s; coverage not measured | Partly |
| UX | Simple/Advanced split shipped; no user testing | Partly |
| Code and repo hygiene | Clean core; repo carries unrelated documents | Partly |

---

## 1. Core value

**Measured:** the causal layer produces a supported explanation for 106 of 596
real incidents (17.8%). 36% of incidents are too short to test at the default
Granger lag. Every number comes from one Windows 11 Home laptop. Correctness
has been shown on injected faults only, and in every passing injection the
cause was also the loudest metric.

A product has to answer "what does the user get on the other 82% of
incidents?" Today the answer is an anomaly flag and a severity ranking with an
honest note that no causal chain was found. That is defensible for research.
For a product it is the main gap.

**Minimum bar:**
- Evaluate on at least three more machines of different hardware and usage.
- Run at least one injected fault whose cause is not the top-severity metric.
- Decide the default Granger lag. At `L=3` nothing falls below the sample floor
  (0 vs 215 incidents), explanations are 110 vs 106, and every incident gets a
  real "tested" answer instead of "not tested".
- Make the non-causal answer useful in its own right: the top anomalous
  metrics, the processes active at the time, and matching Event Log entries,
  described in plain language.

**Overkill now:** fleet-level learning, sharing models across machines.

## 2. Collection reliability

**Measured on this machine:** 85.5% coverage over the last 30 days, median
unbroken run 7 hours, longest 91 hours. This is far better than the 27.8%
recorded before the supervisor existed.

**Found during this review:** the collector had been dead for about an hour
and nothing noticed. The rebuild killed it, and the supervisor script exits by
design when the collector's `.exe` is missing (`schedule.py:289`, "uninstalled:
leave"). A rebuild briefly deletes the `.exe`, so the supervisor left and never
came back. A user who moves the extracted folder, or unzips a new version over
the old one, will hit the same path: `supervise.ps1` hard-codes the absolute
path of the install (`D:\vscode\majorprojectt\dist\...` on this machine).
Collection then stops until the next logon, or indefinitely if the path is
gone. I restarted it; it is collecting again.

The UI does not warn when the newest sample is old. It reports "Collecting
every 30 seconds" whether or not anything has been written recently.

**Minimum bar:**
- Show "Collection stopped N minutes ago" in the status sentence when the last
  sample is more than a few minutes old, with a Restart button.
- Install to a fixed per-user location (see §4) so the supervisor path cannot
  go stale.
- Record why collection stopped (clean exit, crash, sleep, killed).
- Run a 7-day soak on a second machine and watch memory growth and coverage.

**Overkill now:** a Windows service running as SYSTEM. It needs admin rights
and conflicts with the per-user privacy model.

## 3. Security and privacy

**Done well:** no network code in the collector; tiered retention (foreground
app 30 days); consent enforced at two layers; `secure_delete` on; model
encrypted with DPAPI under the user's credentials; atomic writes; pause
control; `delete-all-data`. Two rounds of review found and fixed real defects.

**Gaps:**
- The 227 MB database is plaintext on Windows Home, which lacks EFS. Most
  consumer laptops run Home.
- The model on this machine is still plaintext until it is retrained.
- No written privacy policy or data-flow statement a non-technical user can
  read. The README covers it, but users don't read READMEs.
- Dependencies are not scanned for known vulnerabilities.

**Minimum bar:**
- Encrypt the database independently of EFS, e.g. SQLCipher, with the key
  wrapped by DPAPI.
- Encrypt the model on first launch after upgrade, not only on retraining.
- A one-page privacy statement shown in the app and on the release page.
- `pip-audit` in the release checklist.

**Overkill now:** a formal threat model document, penetration testing, SOC 2.

## 4. Packaging and install

**Measured:** the v1.5.1 download is 1.99 GB, up from 272 MB for v1.5.0. The
build machine has the CUDA build of PyTorch (`torch==2.12.0+cu130`), so 2.6 GB
of NVIDIA libraries were bundled (`cublasLt64_13.dll` alone is 456 MB). The
app does not use the GPU. `requirements.txt` names the CPU index, but the
build used whatever was installed.

Distribution is a ZIP plus `Install LocalRCA.bat`. The binaries are
self-signed, so SmartScreen warns every new user.

**Minimum bar:**
- Build in a clean virtual environment with the CPU-only PyTorch wheel. This
  alone should bring the download back to roughly 300 MB.
- Consider exporting the trained LSTM to ONNX and running inference with
  ONNX Runtime, keeping PyTorch only for training. This is the largest single
  size reduction available, but needs design work because training also
  happens on the user's machine.
- Replace the ZIP with a per-user installer (Inno Setup or MSIX): fixed
  install path, Start menu entry, clean uninstall with an "also delete my
  data?" prompt, and no admin rights needed.
- A code-signing certificate from a trusted authority (an OV certificate or
  Azure Trusted Signing). Check current eligibility and pricing; both change.

**Overkill now:** an MSI for Group Policy deployment, multiple architectures.

## 5. Updates

**State:** an opt-in check that reads the latest GitHub release tag and tells
the user. It downloads nothing.

**Minimum bar:** in-app "download and install" that fetches the signed
installer, verifies its signature and SHA-256, and runs it. Keep it opt-in to
match the no-egress promise.

**Overkill now:** delta updates, staged rollouts, update channels.

## 6. Supportability

**State:** unhandled exceptions go to `desktop.log`. Nothing reaches the
developer, by design.

**Minimum bar:**
- An "Export diagnostics" button that zips the logs, version, OS edition,
  coverage figures and settings (no telemetry rows) for the user to attach to
  a bug report.
- A GitHub issue template that asks for that zip.

**Overkill now:** automatic crash reporting. It would break the no-egress
promise, and that trade should only be made deliberately.

## 7. Build and release engineering

**State:** every release is built by hand on this laptop. No `.github/`
directory, no CI. `requirements.txt` uses `>=` everywhere and mixes build and
test tools (`pytest-qt`, `pyinstaller`) into runtime dependencies. There is
no lock file, so two builds a week apart can ship different libraries. The
PyTorch size regression is exactly that failure.

**Minimum bar:**
- A GitHub Actions workflow on a Windows runner: lint, test, build, and upload
  the artifact on every push to `main`. Release builds happen only from CI, on
  a tag.
- Pin exact versions in a lock file (`pip-tools` or `uv`) and split runtime
  from dev dependencies.
- Keep the signing key in CI secrets, not in the working tree.

**Overkill now:** multiple environments, deployment pipelines, DORA metrics.

## 8. Testing

**Measured:** 150 tests, all passing, 2 min 52 s. The slowest is a 21 s
desktop smoke test. Coverage is not instrumented, so it is unknown.

The project's own history shows the tests' blind spot: both review rounds
found real defects in code whose tests all passed (nine in the first round,
the plaintext checkpoint in the second). The tests check the success path and
single functions; the defects were in failure paths and in how processes
interact.

**Minimum bar:**
- Measure coverage (`pytest-cov`) and publish the number. Don't set a target
  until you've seen it.
- A schema-migration test: open a database written by the previous release
  with the new build. `SCHEMA_VERSION` is 1 and the upgrade path has never run.
- An end-to-end test on a clean Windows VM in CI: install, consent, collect
  for a few minutes, uninstall.

**Overkill now:** mutation testing, property-based testing everywhere, a
dedicated QA environment.

## 9. UX

**Done:** default and advanced views, plain status sentences, consent dialog,
pause, contrast fixes, visible focus.

**Gaps:**
- The first 21 hours are dead time. The app says so but offers nothing to do.
- Nothing tells the user an incident happened. They have to open the app and
  press Find Incidents. The tool only helps people who already suspect
  something went wrong.
- No user testing. The default view is a hypothesis.
- DPI scaling above 100%, keyboard-only use and screen readers are untested.

**Minimum bar:**
- A tray icon with a notification when a significant incident is detected.
- A demo dataset so a new user can explore the UI on day one.
- Five people who aren't developers each try it for a week, and you watch
  what confuses them.

**Overkill now:** localisation, theming, a plugin system.

## 10. Code and repo hygiene

*Cleanup done after this assessment: the files listed below were removed or
moved, except `INTERVIEW_PREP.md`, which is kept on purpose.*

**Clean:** one engine module shared by the UI and tools, lint clean, clear
separation of collector and desktop processes.

**Remove from the product repo:**
- `PRD.md` (3,516 lines) describes a different product: RCA for production
  microservice failures.
- `docs/INTERVIEW_PREP.md`, `docs/RESUME_MATERIAL.md`, `docs/TEAM_OVERVIEW.md`
  and the `.docx` drafts in `docs/`. These are personal and academic material,
  not product documentation.
- `docs/Repository_Overview.md` and `docs/UI_overview.md` (17 lines each) and
  `PRODUCTION_CHECKLIST.md`, which this document supersedes.
- `src/train_and_run.py`, which nothing imports. Move it to `tools/` if it's
  still used by hand.

Keep the paper and report, but in their own folder or repo, so product and
coursework don't share a release history.

---

## Order of work

Ordered by cost against consequence: cheap fixes that protect everything else
come first.

| # | Item | Effort | Why now |
|---|---|---|---|
| 1 | Stale-collection warning in the UI | Hours | Collection just failed silently for an hour |
| 2 | CPU-only PyTorch in a clean venv, rebuild v1.5.2 | Hours | Download is 7x larger than it needs to be |
| 3 | Pin dependencies; split runtime and dev | Hours | Caused #2 |
| 4 | GitHub Actions: lint, test, build on Windows | 1 day | Every release is currently unreproducible |
| 5 | Repo cleanup (§10) | Hours | Cheap, makes everything else easier to read |
| 6 | Diagnostics export button | 1 day | Only way to learn about failures in the field |
| 7 | Per-user installer with fixed path | 2–3 days | Removes the supervisor path bug for good |
| 8 | Encrypt the database (SQLCipher + DPAPI key) | 2–3 days | Closes the Windows Home gap |
| 9 | Tray notifications on incident | 2–3 days | Turns a tool you visit into one that tells you |
| 10 | Second-machine evaluation, non-loudest-cause fault | 1–2 weeks | Core value is unproven beyond one laptop |
| 11 | Trusted code-signing certificate | Money, not time | Removes the SmartScreen warning |
| 12 | Five-person user trial | 1–2 weeks | The default view is untested with real users |

Items 1–5 are about a week of work and move four areas from "no" to "partly".
Items 10 and 12 decide whether this is a product at all. They can run in
parallel with everything else, and should start first because they take
longest.
