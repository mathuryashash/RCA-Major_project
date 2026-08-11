# Production Readiness Checklist

What stands between LocalRCA today and something you would hand to a stranger
without caveats. Ordered by what would embarrass the project first.

`✅ done` · `⚠️ partial` · `❌ missing`

---

## P0 — Blocks calling this a product

### ❌ Collection coverage and collector supervision

**Measured: 27.8% coverage over 13.2 days, median unbroken segment 8.5 minutes.**

The collector is a Startup-folder entry with nothing watching it. If it dies
mid-session it stays dead until the next logon. Everything downstream inherits
this: fragmented training data, unanalysable incident windows, unstable drift
readings.

*This is the single highest-value fix in the project.* Nothing about the model
matters until the data underneath it is continuous.

- [ ] Restart the collector automatically when it dies (watchdog, or a
      Windows Service, or a scheduled task with restart-on-failure)
- [ ] Record why it stopped — clean exit, crash, or machine sleep
- [ ] Surface coverage prominently in the UI, not buried in Captured Data
- [ ] Target: **>90% coverage, median segment measured in hours**

### ❌ Evaluation harness

There is no ground truth, so there is **no measured precision or recall
anywhere in this project**. Every correctness claim is a plausibility
judgement.

- [ ] Inject known faults — CPU stressor, disk filler, memory hog — and assert
      they are detected
- [ ] Assert the correct process is attributed
- [ ] Measure the false-positive rate on an idle machine overnight
- [ ] Track causal yield: what fraction of runs produce ≥1 surviving edge

Until this exists, "is the root cause correct?" has no answer.

### ⚠️ Long-run stability

Never run unattended for more than ~2 minutes under observation.

- [ ] 24-hour soak: does memory grow? RAM was seen at 444 MB and later 670 MB
      in separate observations, and that gap was never explained
- [ ] Does the collector survive sleep, hibernate, dock/undock, user switch?
- [ ] Does the DB stay healthy across an unclean shutdown?

---

## P1 — Needed before wider distribution

### ⚠️ Resource footprint

**Measured on this machine:** collector at **0.78% of total CPU** (28 logical
cores — so roughly a fifth of one core), **27 MB RAM**, database growing
**~2.2 MB/day**.

RAM and disk are fine. The CPU figure is higher than a background sampler
should need and deserves a proper look — the 90-second window may have caught
a process-sampling burst rather than steady state.

- [ ] Profile over an hour, separating idle cadence from burst sampling
- [ ] Throttle or pause on battery
- [ ] Add a visible pause control — there is currently no way to stop
      collection short of uninstalling
- [ ] Cap database growth, or make retention configurable

### ❌ Update mechanism

No way to ship a fix. A user with a broken version stays on it.

- [ ] Check for updates, or at minimum notify that a newer version exists
- [ ] Ensure extracting a new release over an old one is safe (it currently
      breaks the Start menu shortcut, which the app now repairs at launch)

### ❌ Crash reporting reaching the developer

`desktop.log` captures unhandled exceptions, but only on the user's disk. If
nobody sends it, nobody knows.

- [ ] A one-click "export diagnostics" bundling logs, version and coverage
- [ ] Deliberate decision on whether opt-in error reporting is compatible with
      the no-egress promise — **it probably is not**, and that trade-off should
      be made explicitly rather than by omission

### ⚠️ Schema migration

`SCHEMA_VERSION` exists and columns are added defensively, but the upgrade
path has never been exercised across a real version change.

- [ ] Test: v1 database opened by a v2 build
- [ ] Decide what happens when a model was trained on features that no longer
      exist

### ❌ Code signing

Unsigned, so SmartScreen warns every user. Currently mitigated only by the
published SHA256.

- [ ] Sign, or accept permanently and document plainly (current choice)

---

## P2 — Quality and polish

### ⚠️ UI robustness

- [x] Empty states on the figure panels
- [x] Database read errors surfaced rather than freezing stale numbers
- [x] Progress reporting on both long operations
- [x] Full-screen figures
- [ ] **DPI scaling** — never tested at 125%/150%/200%
- [ ] **Small screens** — window opens at 1400×900; behaviour below that is unknown
- [ ] **Multi-monitor** — full-screen dialog on the correct display
- [ ] **Keyboard navigation** — tab order, Enter to activate, Esc to cancel
- [ ] **Screen reader labels** on controls and figures
- [ ] Contrast audit against WCAG AA

### ❌ Settings persistence

Window size, epochs, window size, Granger lag all reset each launch.

- [ ] Persist preferences (`QSettings`)
- [ ] Remember the last analysed range

### ⚠️ First-run experience

- [x] Consent dialog explaining what is recorded
- [x] Install completed without a command line
- [x] First-launch slowness documented (52 s cold, 9.5 s warm)
- [ ] The ~21-hour wait is stated but not *designed for* — the app is close to
      useless in its first day and does not offer anything to do meanwhile
- [ ] No sample or demo dataset to explore the UI before real data exists

### ❌ Uninstall completeness

- [x] Removes the logon entry, Start menu shortcut, Add/Remove entry
- [x] Keeps user data deliberately, and says so
- [ ] Does not offer to remove the data during uninstall — a user removing the
      app has no obvious prompt to erase what it collected
- [ ] The extracted folder must still be deleted by hand

---

## P3 — Would be expected of a mature product

- [ ] Installer (MSI/NSIS) rather than a ZIP
- [ ] Portable mode that writes nothing outside its own folder
- [ ] Localisation — currently English only, hardcoded
- [ ] Per-user isolation verified on a shared machine
- [ ] Report templates or export to PDF/HTML
- [ ] Comparison across time ranges
- [ ] Smaller footprint (1,110 MB; PyTorch is most of it)

---

## Verification checks to run before any release

A concrete list you can execute.

### Build and package
- [ ] `python -m pytest tests -q` — all pass
- [ ] `.\packaging\build.ps1` — exits 0, both exes present
- [ ] `.\packaging\make_release.ps1` — ZIP and SHA256 produced
- [ ] Extract the ZIP to a **clean folder** and run from there
- [ ] Verify the published SHA256 matches the file

### Fresh-machine simulation
- [ ] Rename `%LOCALAPPDATA%\RCA` aside, launch, confirm the consent dialog
      appears and that declining collects nothing
- [ ] Accept, confirm logon entry, Start menu shortcut and Add/Remove entry
      all appear
- [ ] Confirm the app is usable with no data — no crashes, no blank panels,
      readiness clearly explained
- [ ] `uninstall`, confirm all three registrations are gone and data remains
- [ ] `delete-all-data`, confirm the directory and temp charts are gone

### Functional
- [ ] Train — completes, progress moves per epoch, model saved
- [ ] Find Incidents — every incident offered can actually be analysed
- [ ] Run RCA — completes, and **runs a second time** after changing settings
- [ ] Both figures render and open full screen
- [ ] Export Markdown and JSON
- [ ] Confirm the report refuses a causal claim when no edge survives

### Stability
- [ ] Leave running 24 h; check memory growth and `desktop.log`
- [ ] Sleep and resume; confirm collection continues and a gap is recorded
- [ ] Reboot; confirm the collector starts at logon

---

## Honest summary

**Solid:** the pipeline works end to end, error handling and logging are real,
consent and uninstall are honest, licensing obligations are met, and the
reporting refuses to overstate its evidence — which is rarer than it sounds.

**The two things that matter most are both absent:** the collector is
unsupervised, so the data underneath everything is 27.8% complete; and there
is no evaluation harness, so nobody can say whether the answers are right.

Everything in P2 and P3 is polish. Those two are the difference between a
project that demonstrates a technique and a tool someone can rely on.
