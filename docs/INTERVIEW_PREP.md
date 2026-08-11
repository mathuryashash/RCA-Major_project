# Interview Preparation

Questions a fresher is realistically asked about this project, with answers
grounded in what the code does and what was measured.

**One piece of advice before the questions.** The strongest thing about this
project in an interview is not the LSTM — it is that you can name what does
not work and why. Interviewers hear "I built an AI system that finds root
causes" constantly. They rarely hear "the causal layer returns nothing on
most real incidents, here is the measurement, here is why, and here is what I
changed as a result." The second answer is the one that sounds like an
engineer.

Do not memorise these. Understand them well enough to be wrong in public
about the parts you are genuinely unsure of.

---

## A. The 60-second summary

**Q: Tell me about your project.**

> LocalRCA is a Windows desktop application that explains why a machine
> slowed down or crashed. A background collector records system metrics every
> 30 seconds, process activity every 5 minutes, and selected Windows Event Log
> entries. An LSTM autoencoder learns that specific machine's normal
> behaviour, so anomalies are relative to *your* machine rather than a generic
> threshold. When something goes wrong, it scores the incident window, runs
> Granger causality between the anomalous metrics to work out which moved
> first and drove the others, and produces a report.
>
> Everything runs locally — no telemetry leaves the machine. It's packaged
> with PyInstaller as two executables, about 93 tests, and I have measurements
> for the runtime characteristics and, importantly, for where it fails.

**Q: Why did you build it?**

> Diagnosing a slow laptop means manually lining up resource graphs against
> the Event Log and guessing. Cloud observability tools assume a fleet and a
> network path. I wanted to see whether the same idea works for one machine,
> offline, with no labelled data.

---

## B. Machine learning

**Q: Why an autoencoder rather than a classifier?**

> No labels. Nobody records "my laptop was slow at 14:32 because of Windows
> Update". A classifier needs examples of each fault; I have none. An
> autoencoder trains only on normal data, learns to reconstruct it, and
> reconstruction error becomes the anomaly score — no anomalous examples
> needed.

**Q: Why LSTM rather than a plain dense autoencoder?**

> A dense autoencoder sees one timestep, so it can't tell a CPU that has been
> at 90% for an hour from one that jumped there a second ago — and that
> difference is the diagnosis. The LSTM takes a window of 12 consecutive
> samples and carries state across them, so it models the shape of behaviour
> over time.

**Q: What's the actual advantage of reconstruction error here?**

> It's available **per metric**, not just per timestep. The model doesn't only
> say "something is wrong" — it says which of the 29 metrics were poorly
> reconstructed. That per-metric score is what the causal stage consumes. A
> model producing a single anomaly score would leave the causal layer with
> nothing to work on. That requirement is why I didn't use Isolation Forest,
> which is honestly a strong candidate otherwise.

**Q: How do you know it works?**

> Honestly — I don't, not in the measured sense. There's no ground truth on a
> personal machine, so I have no precision or recall figures. Every
> correctness claim is a plausibility judgement. The right fix is fault
> injection: run a known CPU or disk stressor and confirm it's detected and
> attributed. I haven't built it, and I'd call that the biggest gap.

*This answer is a strength. Claiming accuracy you cannot measure is the fastest
way to lose credibility.*

**Q: How do you prevent training on anomalous data?**

> Training only uses "clean baseline" segments — periods with no anomalous
> event and no collection gap. Windows never span a gap either, because a
> window bridging a two-hour sleep would teach the model that a discontinuity
> is normal.

**Q: What about model drift?**

> The reference reconstruction error is stored at training time and compared
> to current error on each analysis. Above 2× it's flagged stale. I measured a
> model at 57× against data a week later; retraining brought it to 1.54×.
>
> There's a subtlety I got wrong initially: drift is measured against
> *whichever window you analyse*, so analysing a month-old incident reports
> the model stale even when it's fine. That's a property of the window, not the
> model. Worse, I'd made staleness disable the Run button, so one look at an
> old incident locked the feature entirely. It now warns instead of blocking.

**Q: Why 12 samples for the window?**

> Six minutes — long enough to capture the shape of a spike, short enough that
> real incidents contain a full window. It's adjustable from 6 to 60. The
> trade-off is real: a 60-sample window needs 30 minutes of unbroken
> collection, and my median unbroken segment was 8.5 minutes, so a large
> window disqualifies most of the collected history.

**Q: Overfitting?**

> A validation split, and a minimum of 250 training windows before training is
> allowed — below that it memorises. Stride-5 overlapping windows multiply the
> data available from limited history. That said, with no labelled test set I
> can't measure generalisation properly, only validation loss.

---

## C. Causality — expect the hardest questions here

**Q: What is Granger causality?**

> A statistical test asking whether the past of series X improves prediction
> of series Y beyond Y's own past. If it does, X "Granger-causes" Y.

**Q: Is that real causation?**

> No, and the report says so explicitly. It's predictive precedence. If an
> unobserved factor drives both series, Granger can report an edge between two
> symptoms. It's evidence, not proof, and the UI labels it that way.

**Q: How do you avoid false positives with so many pairs?**

> Three gates. Stationarity first — ADF test with differencing, because
> Granger on a trending series finds structure that's only the trend. Then
> Benjamini-Hochberg FDR correction, because testing 11 metrics is 110
> hypotheses and at p<0.05 you'd expect about 5 false edges by chance. Then an
> effect-size floor using the F-statistic, because with enough samples a
> negligible improvement becomes "significant".

**Q: How often does it actually find a causal chain?**

> Rarely, and that's the most interesting result in the project. On most real
> incidents, zero edges survive.
>
> Two separate reasons. First, a window can be too short to test at all —
> Granger needs `max_lag × 3` aligned samples and differencing costs more, so
> a 16-sample window against a floor of 17 means *no pair is ever compared*.
> The system used to report that identically to "tested, nothing survived",
> which is the difference between a negative result and no result. I now
> distinguish them.
>
> Second, when no edges survive, the ranking degenerates: every metric gets
> identical graph influence and zero outflow, so the score collapses to timing
> and severity. I measured a case where the top two candidates were 0.4620 and
> 0.4616 — a gap of 0.0004 — and the leading candidate changed at every window
> width I tried. The system was presenting an arbitrary ordering with a
> confidence percentage while its own evidence section said no causal claim was
> supported. It now refuses to name a root cause in that situation.

*If you can deliver that answer clearly, it is worth more than any part of the
model architecture.*

**Q: Why not a proper causal discovery method — PC, PCMCI, DoWhy?**

> PCMCI/tigramite is the stronger method and would handle multivariate
> confounding better. I chose against it on dependency weight and because its
> output is harder to explain in a report a non-specialist reads. DoWhy targets
> interventional questions given a known graph — I'm trying to discover the
> graph. If I revisited it with more data, PCMCI is where I'd look.

---

## D. System design

**Q: Why two processes?**

> Collection must continue when the window is closed; a tool that only records
> while you watch it records nothing useful. It also means a GUI crash doesn't
> stop collection.

**Q: Why SQLite?**

> No server, no install step, no admin rights — it ships as a ZIP. WAL mode
> lets the GUI read while the collector writes, which the design depends on.
> The workload is tiny — 10,555 rows after 13 days.

**Q: How do you handle the machine sleeping?**

> Gaps are first-class. Each sample records elapsed time since the previous
> one; above 1.5× the cadence it's a recorded discontinuity. Everything
> downstream works in contiguous segments rather than on the raw table —
> training windows, incident validation, coverage reporting.

**Q: How do you keep the UI responsive?**

> Three QThread workers — training, incident detection, inference —
> communicating by Qt signals. The engine functions take an optional progress
> callback and the worker passes its signal's emit directly, so the engine
> stays UI-agnostic and the CLI passes nothing.

**Q: How is it tested?**

> About 93 tests, headless, roughly a minute. Real SQLite through the real
> schema rather than mocks, stub detectors with caller-supplied flags so run
> segmentation can be tested in isolation, and pytest-qt for the widgets.
> Several tests exist specifically to pin bugs that already happened once.

---

## E. Bugs — have two or three ready

Interviewers learn more from a debugging story than from an architecture
diagram. These are real.

**Q: Tell me about a difficult bug.**

> Stage 2 sat at 0% forever and did nothing. No error on screen.
>
> The worker stored its analysis window as `self.start` — which overwrote
> `QThread.start`, the method that launches the thread, with a pandas
> Timestamp. Clicking Run raised `TypeError: 'Timestamp' object is not
> callable` inside the click handler, so no worker was ever created. Training
> was unaffected because its worker takes no such argument, which made it look
> like an RCA-specific problem rather than a one-line constructor bug.
>
> Two lessons. First, subclassing a framework class means its attribute
> namespace isn't yours. Second — and this is the one I actually took away —
> every timing measurement I'd taken from source was correct and irrelevant,
> because calling the pipeline directly bypasses the worker entirely. I was
> measuring a path the UI doesn't take. I only found it after routing
> unhandled exceptions to a log file; in a windowed build there's no stdout, so
> the traceback had been going nowhere.

**Q: A packaging problem?**

> Training worked from source and failed in the packaged build with
> `No module named 'optree'`.
>
> optree is an optional PyTorch dependency. My excludes list is generated from
> a static import closure, so anything imported lazily gets excluded — but
> PyInstaller still shipped optree's `.dist-info`. PyTorch checks
> `importlib.metadata.version("optree")`, read a version from the bundled
> metadata, concluded the package was present, imported it, and failed inside
> Adam's constructor. Metadata without the module is worse than neither.

**Q: A mistake you made?**

> I extended "delete all data" to erase the whole data directory. The
> collector's `stop.flag` lives *inside* that directory, so the erase retracted
> the stop request microseconds after making it. The collector polls once every
> 30 seconds, never saw it, kept the database locked — and every 250 ms retry
> destroyed the trained model and reports again, while the telemetry, the
> actual privacy concern, survived. Exactly inverted.
>
> It passed its tests. Deletion is now gated on the database unlink succeeding,
> which is proof the collector has exited. What I took from it is that a fix
> can be worse than the bug, and tests that only check the happy path won't
> tell you.

---

## F. Privacy and security

**Q: What data does it collect, and where does it go?**

> System metrics, the names of the busiest processes, and an allowlist of
> Event Log entries. Never window titles, keystrokes or file contents. It all
> stays in `%LOCALAPPDATA%\RCA`. There are no network imports anywhere in the
> source — I verified that rather than asserting it.

**Q: How do you handle consent?**

> A first-run dialog naming what's recorded, at what cadence, what is never
> recorded, retention, and how to erase it. Nothing is collected until the
> user agrees. It used to be a command-line step, which meant anyone who
> didn't read the README was never asked and silently collected nothing.

**Q: Any security considerations?**

> One I fixed: the collector was launched with `shell=True` on a command built
> from the user's profile path. Quoting handles spaces and `&` but `cmd.exe`
> expands `%NAME%` even inside double quotes. Nothing about launching a known
> executable needs a shell, so it passes an argument vector now.
>
> Known gaps: the build is unsigned, and the database is plaintext. The
> plaintext part I'd defend — it's in a per-user directory behind Windows ACLs
> and the threat model isn't "attacker already has your files".

---

## G. Questions you should ask about your own numbers

Be ready for these, because a sharp interviewer will find them.

**Q: Your coverage is 27.8%. Isn't that a broken system?**

> It's the biggest weakness, yes. Partly self-inflicted — I was killing the
> collector constantly during development builds. But the architecture
> contributes: it's a Startup-folder entry with nothing supervising it, so a
> crash means it stays dead until the next logon. Fixing that would improve
> results more than any change to the model, because everything downstream
> inherits it: training data, incident analysability, causal window length.

**Q: Isn't 1.5 GB unreasonable for this?**

> Yes. Most of it is PyTorch, for a model that trains in 8.5 seconds. If I were
> starting again I'd seriously evaluate whether a much smaller detector could
> give me the per-metric error I need, because the deep-learning dependency
> also caused three separate packaging failures.

**Q: What would you do next?**

> Two things, in order. Supervise the collector so coverage goes above 90%.
> Then build a fault-injection harness so "is the root cause correct" becomes
> a measurement instead of an impression. Everything else is secondary to
> those.

---

## H. Rapid-fire

| Question | Answer |
|---|---|
| Language/stack? | Python 3.13, PyTorch, PySide6, SQLite, PyInstaller |
| Lines of code? | ~4,800 in `src/`, plus tests |
| How long? | Several weeks, iteratively |
| Team size? | Solo |
| Sampling rate? | 30 s system, 300 s process, 300 s events |
| Features? | 29 metrics |
| Training time? | 8.5 s default, ~64 s at maximum settings |
| Inference time? | 0.2–12.5 s, grows with the square of window size |
| Cold start? | ~21 hours of clean collection |
| Test count? | ~93, about a minute headless |
| Biggest weakness? | 27.8% collection coverage, and no ground-truth evaluation |

---

## I. Traps

**Don't say "it predicts failures."** It explains ones that already happened.
There is no forecasting.

**Don't say "it uses AI to find the root cause."** It ranks candidates using
a statistical test and says plainly when the evidence doesn't support a
claim.

**Don't claim accuracy figures.** There are none. Saying so is stronger than
inventing them.

**Don't oversell Granger.** If asked whether it proves causation, the answer
is no, and knowing the difference is the point.

**Do volunteer the negative results.** The causal layer usually finding
nothing, and the work done to make the system admit that, is the most
technically interesting part of the project.
