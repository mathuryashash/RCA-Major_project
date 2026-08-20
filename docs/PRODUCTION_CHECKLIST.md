# Production Readiness Checklist

What stands between LocalRCA today and something you would hand to a stranger
without caveats. Ordered by what would embarrass the project first.

`✅ done` · `⚠️ partial` · `❌ missing`

---

## P0 — Blocks calling this a product

### ⚠️ Collection coverage and collector supervision

**Measured before the fix: 27.8% coverage over 13.2 days, median unbroken
segment 8.5 minutes.** A collector that died mid-session stayed dead until the
next logon, and everything downstream inherited it.

- [x] Restart the collector when it dies. Task Scheduler is the proper
      mechanism and needs elevation — measured, the COM API returns
      `E_ACCESSDENIED` as a standard user — so the logon entry starts a hidden
      PowerShell supervisor instead, backing off and giving up after twelve
      attempts
- [x] Verified: killed the collector, supervisor restarted it within 45s
- [x] The sampling loop survives tick failures and stops after 20 consecutive
      ones rather than logging forever
- [ ] **Re-measure coverage over a fresh multi-day window.** The 27.8% figure
      predates supervision *and* was depressed by rebuilds killing the
      collector, so it cannot be compared directly
- [ ] Record why it stopped — clean exit, crash, or sleep
- [ ] Surface coverage prominently in the UI
- [ ] Target: **>90% coverage, median segment measured in hours**

### ⚠️ Evaluation harness

`tools/evaluate_detection.py` causes a known disturbance and scores what the
pipeline saw. **First measured results on this machine:**

| Run | Samples | Flagged | Causality | Result |
|---|---|---|---|---|
| CPU burn, 7 min | 14 | 6 of 29, correct metrics present | **never tested** — below the Granger floor | PASS (detection only) |
| CPU burn, 30 min | 60 | 6 of 29 | **6 edges survived** of 10 significant pairs | **PASS** |
| Disk burn, 30 min | 60 | 4 of 29 | 1 pair accepted, **pruned by topology** | PASS, unexplained |
| Memory hold, 30 min | 60 | 2 of 29 | no chain | **FAIL — wrong culprit named** |
| Memory hold, 30 min, clean baseline | 60 | **0 of 29** | — | **not measurable on this host** |
| Idle, 30 min | 60 | 1 of 29 (`mem_available_mb`) | no chain | 3.4% false positives |

**The 30-minute run is the one that settles it.** We caused a CPU burn, and
the ranking named it:

```
1. cpu_pct            score 1.000   Critical
2. cpu_pct_max_core   score 0.916   High
3. swap_used_delta    score 0.719   Medium
```

The causal directions are right too — CPU drives the rest, nothing drives CPU:

```
cpu_pct_max_core → disk_busy_pct    lag=2   strength 0.756
cpu_pct_max_core → swap_used_delta  lag=3   strength 0.262
swap_used_delta  → disk_busy_pct    lag=2   strength 0.436
```

Same fault and same code at 7 and 30 minutes, differing only in duration:
**the causal layer was never broken, it was starved.** It also correctly
reported "not tested, window too short" at 7 minutes instead of inventing an
answer, which is the reporting change from earlier doing its job.

**The disk run is the more instructive failure.** Detection passed —
`disk_write_bps` and `disk_busy_pct` flagged, load attributed to `python.exe` —
and `disk_write_bps` ranked first at 1.000. But the graph was empty. Exactly
one pair passed both statistical gates:

```
net_sent_bps → cpu_pct_max_core     p=0.0033   lag=1   strength 0.136
```

and the subsystem map has no network→CPU path, so it was pruned. With no edges
the score collapses to severity alone — so the right answer arrived by a route
the system cannot claim as causal, and the report says so.

That run also caught a **reporting defect, now fixed**: a topology-pruned pair
was described as "no edge survived multiple-testing correction", blaming the
statistics for a decision the map made. The two are now distinguished.

**The memory run failed, and the failure was in production, not the test.**
A process held 1.15 GB while sleeping. Detection worked; attribution named
`SearchIndexer.exe`, `WmiPrvSE.exe`, `Taskmgr.exe` and `MsMpEng.exe`, and the
process actually responsible never appeared. `load_process_attribution` ordered
by `avg_cpu_pct` — `max_rss_bytes` was selected and never sorted on — so a
memory-bound cause could not be named **in any incident, ever**. Fixed by
ranking on both; verified against the recorded window, where `python.exe` now
appears at 1,537 MB.

**The harness was also wrong, and worse.** It printed `ATTRIBUTED to us: no`
and returned PASS, because only detection gated the verdict. The line below
was ticked on the strength of a check that never ran. It runs now.

- [x] Inject a known fault and assert it is detected
- [x] Assert the correct process is attributed — **now actually enforced**;
      previously printed and ignored, which is how a failing run scored PASS
- [x] Measure the false-positive rate at rest
- [x] **Assert the ranking names the injected cause** — `cpu_pct` first at 1.000
- [x] Memory fault — run at 1.15 GB held for 30 minutes, bounded deliberately
- [x] **Re-run memory end to end with attribution fixed** — fresh 30-minute
      injection of 0.78 GB, detected and attributed, `ATTRIBUTED to us: yes`
      under the newly-enforced gate. The re-run also caught the fix being
      incomplete: the culprit was named *tenth of ten* because ordering still
      used CPU. Processes are now ranked by their share of the window maximum
      in either dimension, which moves it to second
- [x] **Explain why `mem_pct` does not flag under a memory injection** —
      answered, and the answer is that this machine cannot test it. Its median
      `mem_pct` across 30,199 samples is **95.2%**; a 1.91 GB hold on a cleared
      machine landed the window at 87.4%, the **9th percentile** of its own
      history. The injection is bounded to half of free memory for safety, and
      on a habitually-full host that bound cannot exceed normal usage — the
      fault is undetectable by construction. The harness now reports
      INCONCLUSIVE with the arithmetic instead of blaming the detector
- [ ] **Re-run memory detection on a second machine.** This is now the
      strongest reason to test other hardware: not generality, but a
      measurement that is unobtainable here at all
- [ ] Tighten the harness's attribution check — it matches any process named
      "python", and this machine runs several
- [x] **Track causal yield across many incidents** — 175 incidents found in
      real history, 92 analysable, surveyed in 1.7 min by
      `tools/measure_causal_yield.py`. **31.5% produce a supported chain**;
      **47% of all incidents are below the Granger floor** and cannot be
      tested at all; yield doubles from 25% to ~53% once windows exceed an
      hour. Repeatable, no injection required
- [ ] Repeat on a second machine — every number here is from one host
- [ ] Test a fault whose cause is *not* the top-ranked metric. **Both passing
      runs put the injected fault at the top of a severity ranking, so nothing
      here separates a correct causal answer from a correct severity answer**
- [x] **Audit what the subsystem map rejects** — `tools/audit_topology_map.py`.
      Of 115 accepted pairs, 65 survive, **30 (26%) are rejected by the map**
      and 20 (17%) by cycle-breaking. (An earlier figure of 43% attributed
      both to the map; cycles are broken before the map is consulted.)
- [ ] **Decide whether to give the map reverse edges.** It is a strict total
      order `power, process → cpu → memory → disk → network`, so `network` is
      a sink and nothing can ever be reported as caused by network activity.
      Three of the five rejected classes describe real mechanisms —
      `network → disk` (a download writes to disk; **strength 0.982, the
      strongest relationship measured anywhere in this project**),
      `disk → memory` (standby file cache), `disk → cpu` (interrupt and DPC
      handling). Adding them makes the graph cyclic, and cycle-breaking
      already removes 17% of pairs, so the two filters would start to fight

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
**3.33 MB/day** (re-measured over 18.2 days; the earlier 2.2 MB/day figure came
from a shorter, quieter window and understated the rate by 51%).

RAM and disk are fine. The CPU figure is higher than a background sampler
should need and deserves a proper look — the 90-second window may have caught
a process-sampling burst rather than steady state.

- [ ] Profile over an hour, separating idle cadence from burst sampling
- [ ] Throttle or pause on battery
- [x] **Add a visible pause control** — a tool that records which
      applications you use every 30 seconds had no off switch short of
      uninstalling, which is not a defensible position for software whose case
      rests on privacy. The mechanism already existed: collector and
      supervisor both poll a stop flag, it had simply never been offered to
      the person being recorded. Verified against the live system, not mocks —
      collector 288624 exited on the flag, `resume_collection()` cleared it and
      relaunched under the supervisor as 201120. Resuming deliberately restores
      *supervision*, not just the collector: pausing ends the supervisor loop
      too, so a naive resume would leave collection unsupervised until the next
      logon
- [x] **Cap database growth** — metric history now expires at 365 days
      (matching the event window, so an Event Log fault always has telemetry
      left to explain it), and `VACUUM` returns freed space to the filesystem
- [ ] **Observe a purge actually running.** Measured at the time of the change:
      71.8 MB, **0.0 MB reclaimable, 0 rows past retention** — the install is
      21 days old and the shortest window is 30 days, so the live path has
      never fired. Unit-tested (the test asserts the file on disk halves), not
      yet seen in production
- [x] **Shorten retention on the focus record** — `foreground_app`, sampled
      every 30 s and paired with `user_idle_sec`, reconstructs when the machine
      was in use and roughly what for. It was the most personal field collected
      and had the least protection: on by default, outside the opt-in that
      governs Event Log text, and kept forever. Now erased at 30 days as a
      *column*, so the readings beside it survive and nothing the model trains
      on is touched
- [x] **Say so in the consent dialog** — it listed what is recorded every 30
      seconds and omitted the focus log entirely
- [ ] Make retention user-editable. It is now *shown* in the Captured Data tab
      but changing it needs settings persistence, which does not exist yet

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
- [x] **Small screens** — the window declared a 1024×640 minimum it could not
      render: Stage 2 wanted 1168px of height, the Captured Data table 1752px
      of width, and no view scrolled, so content was clipped with no way to
      reach it. Every tab is now in a scroll area and unwrapped labels no
      longer set the floor. Window minimum: **1155×180 → 532×180**
- [ ] **Multi-monitor** — full-screen dialog on the correct display
- [x] **Focus indicators on every focusable control** — a stylesheet
      background stops Qt drawing the native focus rectangle, so styling a
      widget silently removes its indicator. Verified by rendering, not by
      reading the sheet: 8 of 8 controls now change pixels on focus with no
      size shift, and the test fails if either rule is removed
- [ ] **Keyboard navigation** — tab order, Enter to activate, Esc to cancel
- [ ] **Screen reader labels** on controls and figures
- [x] **Contrast audit** — text was never the problem (body 14.64:1); the
      structure around it was. `BORDER` rendered at **1.32:1** and is now
      **3.07:1**, clearing SC 1.4.11 for non-text contrast, with gridlines and
      row striping raised for readability. Striping deliberately claims no
      compliance: it is decorative and a table reads without it. Pinned by a
      test that recomputes the ratio from the palette

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

**Both P0 items have moved**, though neither is finished:

- The collector is supervised now and demonstrably restarts after a kill. What
  remains is measuring whether coverage actually rises over a fresh multi-day
  window — the 27.8% figure predates the fix and cannot be compared to it.
- Detection is measured for the first time: 6 of 29 metrics flagged under a
  CPU burn with the right ones present and the load correctly attributed,
  against 1 of 29 at rest.

**The ranking is now evidenced once.** A 30-minute CPU burn produced six
surviving causal edges and put `cpu_pct` first at a score of 1.000, with the
causal directions pointing away from CPU rather than toward it. That is the
first end-to-end demonstration that the pipeline identifies a cause it was
never told about.

**One success is not a measurement.** It is a single fault, of a single kind,
on a single machine, where the answer happened to be the most obvious metric.
The harder tests — a fault whose cause is not the loudest signal, and repeats
across machines — remain undone, and until those exist the honest claim is
"demonstrated once", not "validated".

Everything in P2 and P3 is polish by comparison.
