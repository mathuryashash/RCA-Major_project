# LocalRCA — Desktop UI Review

Reviewed at v1.2.1 against `src/desktop/**`, `src/pipeline/visualizations.py`,
`docs/UI_overview.md` and the three captures in `docs/screenshots/`.

Contrast ratios below are computed from the literal hex values in
`src/desktop/theme.py` and `src/pipeline/visualizations.py` using the WCAG 2.1
relative-luminance formula. Alpha colours are composited against their actual
parent surface before measuring.

| # | Area | Score |
|---|------|-------|
| 1 | Visual hierarchy | **4 / 10** |
| 2 | Information density and layout | **4 / 10** |
| 3 | Typography and colour | **6 / 10** |
| 4 | Accessibility | **3 / 10** |
| 5 | DPI and window scaling | **5 / 10** |
| 6 | State communication | **5 / 10** |
| 7 | Honesty in presentation | **4 / 10** |

The headline problem is that the app's honesty lives almost entirely in
`src/reporting/report_generator.py`, and the UI puts that text in the least
prominent widget on the screen while giving the four-decimal score table the
default position. Section 7 covers this; it is the highest-value fix in the
document.

---

## 1. Visual hierarchy — 4 / 10

The app's job is naming a root cause. On the results screen, the root cause is
the least visually emphasised thing present.

**The answer has no typographic weight.** `src/desktop/views/stage2_view.py:279`
writes every cell — rank, metric name, score, confidence — as a bare
`QTableWidgetItem`. The metric name, which is the entire output of the product,
renders at the same 11pt regular weight as the string `"—"` in the Downstream
column. Nothing in the UI is larger or heavier than anything else inside the
results panel.

**The two most prominent controls on the results screen are the two least
important actions.** In `docs/screenshots/stage2-rca-inference.png`, the visually
loudest elements are the full-width, accent-filled "Export Markdown Report" and
"Export JSON Report" buttons (`stage2_view.py:126-133`). They are filled with
`ACCENT_DEEP` at full width and full saturation, while the primary action, "Run
RCA on Collected Telemetry" (`stage2_view.py:72-75`), is disabled and therefore
renders as a transparent ghost outline at 2.96:1 contrast
(`theme.py:113-117`). The eye lands on Export first, Run RCA last. Same
inversion on the Captured Data tab: `data_view.py:100-102` gives "Refresh" — a
button that fires automatically every 30 s anyway (`data_view.py:105-107`) — a
full-width accent slab, making it the loudest element on a screen whose point is
the channel table.

**Every button is the same button.** `theme.py:87-95` gives `QPushButton` an
accent fill, and `theme.py:105-111` makes `#primaryAction` a slightly brighter
accent fill. The difference between "run a 60-second analysis" and "refresh a
table" is a shade of teal and 3px of padding. The `taste-design` rule that
secondary actions should be ghost/outline against a single filled primary is not
applied; the result is that the accent no longer marks importance, only
clickability — which contradicts the stated intent in `theme.py:1-6` ("Cyan marks
anything the user can act on") only in the sense that the intent itself does not
distinguish rank.

**Layout order buries the result.** In `stage2_view.py:28-133` the vertical order
is config box → warning → run button → progress → status → results → exports.
Configuration occupies the top third of the screen permanently, including after
an analysis has completed. The user reads the inputs before the answer on every
single visit.

**What is good:** the compact single-row header (`main_window.py:72-81`,
`theme.py:201-215`) is a genuine improvement over a hero block, and the tab
labels "1 —" / "2 —" encode workflow order clearly.

---

## 2. Information density and layout — 4 / 10

**The empty lower area is real and is an empty-state problem, not a sizing
problem.** `stage2_view.py:273` does `setRowCount(len(root_causes))`. With zero
candidates — or before any run — the panel is a 500px void with a header row
floating at the top, exactly as captured in
`docs/screenshots/stage2-rca-inference.png`. The Causal Graph and Anomaly
Timeline tabs both got placeholders (`stage2_view.py:116-121`); the Root Causes
tab, which is the default tab, did not. Stretching the columns
(`stage2_view.py:86-88`) widened the header but did nothing about the void
below it, because the void is rows, not columns.

Note also that `header.setSectionResizeMode(QHeaderView.Stretch)` followed by
`header.setStretchLastSection(True)` on `stage2_view.py:87-88` is contradictory —
`stretchLastSection` has no effect while the resize mode is `Stretch`. The second
line is dead.

**Stage 1 is roughly 45% empty log console.** `stage1_view.py:85-87` gives
`log_console` `stretch=1` — it is the only stretching widget in the layout, so it
absorbs all slack. `docs/screenshots/stage1-baseline-training.png` shows ~480px
of empty black console on a 950px window before training has ever run. The log is
useful *during* a 2–10 minute training run and worthless before and after it.

**The Captured Data table has a 1000px gulf between label and value.**
`data_view.py:95` and `data_view.py:177-178` call `resizeColumnsToContents()` and
then stretch column 1 ("Channel"). Column 1 gets all the slack, so in
`docs/screenshots/captured-data.png` the string `Utilisation` sits at x≈145 and
its value `41.10` at x≈1330, with nothing in between. Scanning a row requires
crossing an empty 1100px field. Stretching column 1 is the wrong choice: the
longest string in the table is `"Frequency ratio (throttle proxy)"` at ~230px.

**The config form is a full-width row per field.** `stage2_view.py:30-63` uses a
single `QFormLayout` in a full-width group box, so `QSpinBox` "Granger Max Lag"
— a two-digit integer — gets a 1600px-wide input
(`docs/screenshots/stage2-rca-inference.png`). Same for the two
`QDateTimeEdit`s. Five short fields consume 260px of vertical space to display
about forty characters of information.

**Nothing is scrollable.** No `QScrollArea` exists anywhere in `src/desktop/`.
When a tab's content exceeds the viewport — which it does, see §5 — controls are
compressed rather than scrolled, and `data_view.py:94`'s hard
`setMinimumHeight(430)` means the compression falls entirely on the
"Collected Store" form above it.

**What is good:** giving `results_tabs` `stretch=1` (`stage2_view.py:125`) is the
right call, and the group-box grouping of the Stage 1 status readouts
(`stage1_view.py:47-60`) is clean and readable.

---

## 3. Typography and colour — 6 / 10

### Type scale

Five sizes inside a 4.5pt span: 14pt hero (`theme.py:205`), 11pt body
(`theme.py:34`), 10pt mono (`theme.py:192`), 9.5pt group-box title and hero
subtitle (`theme.py:82`, `theme.py:213`), plus 16px and 9px inside the Plotly
figures (`visualizations.py:81`, `visualizations.py:50`). 9.5pt and 10pt and 11pt
are not distinguishable at reading distance, so the scale carries no hierarchy —
it only carries noise. Meanwhile there is no step *above* 14pt at all, so the
result of the analysis can never be typographically larger than the window
header.

Numbers are set in the proportional UI font. The Score column
(`stage2_view.py:275`) renders `0.8123` / `0.7940` / `0.7891` with no decimal
alignment and no right-alignment, so a column of scores does not read as a
column. `theme.py:191` already declares a mono stack for `QPlainTextEdit`;
nothing uses it in the table.

### Colour — contrast measurements

Text contrast is mostly strong:

| Pair | Ratio | Verdict |
|---|---|---|
| `TEXT #e6edf3` on `BG #0d1117` | **16.02:1** | AAA |
| `TEXT` on `SURFACE #161b22` | **14.64:1** | AAA |
| `TEXT_MUTED #8b949e` on `BG` | **6.15:1** | AA |
| `TEXT_MUTED` on `SURFACE_ALT #1c2128` | **5.26:1** | AA |
| `ACCENT #2dd4bf` on `BG` | **10.17:1** | AAA |
| `WARNING #f59e0b` on `SURFACE_ALT` | **7.54:1** | AAA |
| `DANGER #f87171` on `BG` | **6.84:1** | AA |
| `TEXT` on `ACCENT_DEEP #0f766e` (button label, `theme.py:87-95`) | **4.63:1** | AA, marginal |
| `#06201d` on `ACCENT` (primaryAction, `theme.py:105-111`) | **9.16:1** | AAA |
| **`#57606a` disabled label on `BG`** (`theme.py:101`, `theme.py:115`) | **2.96:1** | **fails** |
| **`#57606a` disabled label on `SURFACE`** | **2.71:1** | **fails** |

Non-text contrast is where this falls down. WCAG 2.1 SC 1.4.11 requires 3:1 for
the visual boundary of a control where that boundary identifies the control:

| Boundary | Ratio | Verdict |
|---|---|---|
| `BORDER rgba(240,246,252,0.10)` on `BG` (`theme.py:17`) | **1.28:1** | fails 1.4.11 |
| Same on `SURFACE` — input borders, `theme.py:139` | **1.32:1** | fails |
| Disabled button border `rgba(...,0.08)` (`theme.py:102`) | **1.20:1** | fails |
| Table gridline `rgba(...,0.06)` (`theme.py:163`) | **1.17:1** | invisible |
| `alternate-background-color` `SURFACE_ALT` vs `SURFACE` (`theme.py:161-162`) | **1.07:1** | does nothing |
| `SURFACE` tab pane vs `BG` app background (`theme.py:41`) | **1.09:1** | does nothing |
| Progress-bar trough `rgba(...,0.05)` (`theme.py:149`) | **1.11:1** | invisible |

Consequence, visible in all three screenshots: every `QSpinBox`, `QComboBox` and
`QDateTimeEdit` reads as a flat dark band rather than a bordered control, the
alternating table rows are one colour, and the progress bar at 0% is
indistinguishable from a disabled input.

### The blanket `QWidget` background is painting boxes behind every label

`theme.py:30-35` sets `background-color: BG` on the `QWidget` selector. `QLabel`
is a `QWidget`, so every label paints a `#0d1117` rectangle — including labels
sitting inside a `#1c2128` group box. In
`docs/screenshots/captured-data.png` and `stage1-baseline-training.png` this
produces the visible dark rectangles hugging "Clean samples collected", "Current
model" etc., and full-width dark bands across the value column. It reads as an
unintentional striped-table artifact. This one rule is responsible for most of
the "unfinished" feel of the two form-heavy tabs.

### Three colour systems, not one

`theme.py:1-6` documents the deliberate removal of an indigo/purple story that
clashed with a green console. That story is still alive in the figures:

- `visualizations.py:88-90`, `118` — figure background `#151a2e` (indigo-navy).
  Against the app's `#0d1117` this is a **1.10:1** difference in luminance but a
  clearly different hue — the figures sit as blue-tinted rectangles inside a
  neutral-grey app.
- `visualizations.py:33` — causal edges in `rgba(102, 126, 234, ...)`, the exact
  indigo `#667eea` from the removed gradient.
- `visualizations.py:62` — nodes in `#ff4757` / `#ffa502` / `#70a1ff`, none of
  which appear in `theme.py`.
- `visualizations.py:118` and `graph_panel.py:81` — `font-family: Inter`, while
  `theme.py:33` uses `"Segoe UI"` first. `Inter` is not bundled, so on a stock
  Windows machine the figures silently fall back to a *different* font than the
  app chrome.
- `graph_panel.py:80` / `graph_panel.py:117` — hardcoded `#151a2e` and `#7c8aa5`
  duplicated in two places, neither imported from `theme.py`.

`theme.py` also styles none of the object names the code sets:
`figureLegend` (`graph_panel.py:52`, `:153`), `figureTitle` (`graph_panel.py:142`)
and `fullScreenFigure` (`graph_panel.py:137`) have no rules at all. The figure
captions — which are the app's careful explanations of what an arrow means and
what "no arrows" means — therefore render at full 11pt body weight, identical to
everything else, and the full-screen dialog is unstyled.

---

## 4. Accessibility — 3 / 10

### Focus is invisible on every control except three

`theme.py:144` is the only focus rule in the entire stylesheet, and it covers
`QSpinBox`, `QDateTimeEdit` and `QComboBox` only — a 1px border swap from
`rgba(...,0.10)` to `ACCENT_DIM`.

There is **no focus rule for `QPushButton`, `QTabBar::tab`, `QTableWidget`,
`QSlider` or `QCheckBox`.** This is worse than an omission: once a `QPushButton`
has a stylesheet background set (`theme.py:87-95`), Qt stops drawing the native
focus rectangle. So keyboard focus on "Run RCA on Collected Telemetry", "Train
from Clean Collected Telemetry", "Find Incidents", "Full screen", "Refresh", both
Export buttons, and both consent-dialog buttons is **completely undrawn**. A
keyboard user pressing Tab has no way to know where they are. WCAG 2.4.7, failed
outright.

The tab bar is the same: `theme.py:47-64` styles `:hover` and `:selected` but not
`:focus`, so arrow-keying between tabs shows nothing until the selection changes.

### No accessible names anywhere

`grep -rn "setAccessible" src/desktop/` returns nothing. Zero
`setAccessibleName`, zero `setAccessibleDescription` in the application.

Two places lose their names structurally, not just by omission:

- `stage1_view.py:26` — `layout.addRow(label, row)` where `row` is a
  `QHBoxLayout`. `QFormLayout::addRow(QString, QWidget*)` sets the created
  label as the widget's buddy; `addRow(QString, QLayout*)` **cannot**. So
  "LSTM Training Epochs" and "LSTM Window Size (samples)" are visually labelled
  but programmatically anonymous — Narrator announces "5, spin box".
- `stage2_view.py:40` — same pattern, `addRow("Detected incident", incident_row)`.
  The incident `QComboBox`, arguably the most important input in the app, has no
  accessible name.

The rows added with `addRow("Range start", self.start_edit)` etc. *are* correctly
buddied, so the fix is narrowly scoped to those two rows.

### No keyboard shortcuts or mnemonics

No `setShortcut`, no `QShortcut`, no `&` mnemonics on any button.
`main_window.py:88` writes `"1 — Baseline && Training"` — the `&&` correctly
escapes to a literal ampersand, so nothing is broken, but there is also no
`Ctrl+1/2/3` to reach the tabs and no `Alt+R` for Run. Everything is Tab-walking.

### Tab order

Default (creation) order is correct in all three views — no `setTabOrder` is
needed. But `stage2_view.py:134` calls `set_enabled(False)`, which
(`stage2_view.py:141-142`) disables only `refresh_button` and `run_button`. The
combo box, both date-time editors and the lag spin box stay focusable and
editable while Stage 2 is locked, so a keyboard user tabs through four live
controls that cannot lead anywhere, then reaches two skipped buttons, then two
Export buttons that silently do nothing (`stage2_view.py:304`, `:312`).

### Colour-alone meaning

Good news first: the Confidence column (`stage2_view.py:275`) is plain text with
no colour coding, so "Correlation only" vs "Critical" is *not* colour-dependent.
The severity story in the table is text-only and passes.

The failures are in the figures:

- `visualizations.py:62` — node role is encoded as `#ff4757` (root cause) vs
  `#ffa502` (source) vs `#70a1ff` (intermediate). Red-vs-orange is the canonical
  deuteranopia/protanopia confusion pair, and those two colours carry the single
  most important distinction in the graph. Marker size varies (34/26/22,
  `visualizations.py:63`) which helps slightly, but a 34px and a 26px circle are
  not reliably separable across a wide layout.
- `visualizations.py:31-33` — edge strength is encoded as width *and* opacity of
  one colour, with no numeric label. The lag is labelled
  (`visualizations.py:48-51`) at `size=9` in `rgba(180,190,220,0.7)`, which
  composites to **5.21:1** — passes AA, but 9px is below the 12px practical floor
  for a figure a user is expected to read at a glance.
- `visualizations.py:107-113` — the timeline uses Plotly's default qualitative
  palette for up to five metric lines with no dash patterns and no markers.
  Series identity is 100% colour. The red dashed anomaly markers
  (`visualizations.py:112`) are the same red family as one of the default series
  colours.
- `visualizations.py:65` — legend entries are `"🔴 Root Cause"`, `"🟠 Source
  Node"`, `"🔵 Intermediate"`. The emoji is decorative duplication of the swatch,
  a screen reader announces "red circle Root Cause", and per the project's own
  stated reason for stock fonts (`theme.py:8-11`), emoji rendering is exactly the
  kind of platform-dependent glyph the project decided to avoid.

### Consent dialog

`consent.py:50-74` is the strongest screen in the app for accessibility: the
buttons say what they do rather than "OK"/"Cancel" (`consent.py:69-71`), the
disclosure is real HTML in a word-wrapped `QLabel`, and the opt-in checkbox
defaults off. Two gaps: no `setDefault`/`setAutoDefault` so Enter does nothing
predictable, and no `setMinimumHeight` — at ~620px wide the ~40 lines of
disclosure will exceed a 640px-tall minimum window with no scroll area, so on the
smallest supported display the "Start collecting" button can be pushed off-screen
on the one dialog that must never be dismissed blind.

---

## 5. DPI and window scaling — 5 / 10

`main_window.py:23-53` is thoughtfully written and its docstring names the exact
failure it was built to fix. It still has one arithmetic hole, and it lands on the
most common Windows laptop configuration.

Simulating `_size_to_screen` across real configurations (available geometry in
logical px, taskbar deducted):

| Display | Available | Computed | Actual after min-clamp | Bug |
|---|---|---|---|---|
| 1920×1080 @100% | 1920×1032 | 1500×928 | 1500×928 | — |
| 1920×1080 @125% | 1536×826 | 1382×743 | 1382×743 | — |
| **1920×1080 @150%** | **1280×688** | **1152×619** | **1152×640** | **positioned 10px too low** |
| 1920×1080 @175% | 1097×590 | — | maximised | — |
| 1366×768 @100% | 1366×720 | 1229×648 | 1229×648 | — |
| 1366×768 @125% | 1092×576 | — | maximised | — |
| 3840×2160 @200% | 1920×1056 | 1500×950 | 1500×950 | — |

At 1920×1080 @150% — the factory default for most 15" 1080p Windows laptops —
`main_window.py:46` computes `height = 619`, which is below `MINIMUM_SIZE[1]`
(640). `resize()` on line 47 is silently clamped up to 640 by the minimum set on
line 39, but `move()` on lines 50-53 still uses the un-clamped 619. The window is
therefore placed 10px lower than centre and its bottom edge — the status bar and
the Export buttons — sits closer to the taskbar than intended. This is a
milder version of the exact bug the docstring says the function exists to
prevent. Also note the "90% of available" intent quietly fails here: the window
ends up at 93% of available height because of the clamp.

**Fixed pixel geometry that does not survive scaling:**

- `data_view.py:94` — `setMinimumHeight(430)` on the channel table. At
  1920×1080 @150% the window is 640 logical px tall. Subtract the header (~34),
  tab bar (~40), margins (24), the intro label (~40), the nine-row "Collected
  Store" form (~330), the Refresh button (~40) and the status bar (~24), and
  there are roughly **108px** left for a table with a 430px minimum. With no
  `QScrollArea`, Qt resolves this by squeezing the group box above and pushing
  the Refresh button toward or past the bottom edge.
- `theme.py:152` — `QProgressBar { height: 18px }` with `text-align: center` and
  an 11pt font. At 150% the text box grows with the font; the 18px is a hard
  ceiling. Vertical clipping of the percentage text is likely.
- `theme.py:130-135` — `QSlider::handle` at fixed `width: 15px; margin: -6px 0`
  over a `height: 5px` groove. Fixed handle geometry against a fixed groove
  height; at 200% this is a 15px handle on a control the OS expects to be ~30px,
  and it is well under the 44px touch target the guidelines call for on any
  touch-capable Windows device.
- `visualizations.py:87` and `:117` — figures built at fixed `height=520` and
  `height=420`. Inside the results tab at the 1024×640 minimum, the available
  panel height is roughly 150px (see below), so the user sees the top ~30% of a
  520px figure inside a scrolling web view. `graph_panel.py:101-113` already has
  the `fill=True` path that fixes this — but it is only used by the full-screen
  dialog (`graph_panel.py:163`), never by the inline view.

**At the 1024×640 minimum**, Stage 2's fixed chrome is roughly: window margins 24
+ header 34 + tab bar 40 + config group box ~260 + run button 46 + progress 26 +
status 20 + export row 40 + status bar 24 ≈ **514px**, leaving ~126px for
`results_tabs` including its own ~36px tab bar. That is about **90px** of usable
results area — the root-cause table shows its header and one row. The stated
minimum size is not actually a usable size for the app's primary output.

`main.py:164-174` sets no `setHighDpiScaleFactorRoundingPolicy`. Qt 6 defaults to
`PassThrough`, which is the correct choice for fractional scaling, so this is
fine as-is — worth stating explicitly so nobody "fixes" it to `Round` later.

---

## 6. State communication — 5 / 10

**Genuinely good, and worth preserving:**

- The pre-flight cost estimates. `stage1_view.py:96-112` and
  `stage2_view.py:187-211` quote how long an operation will take *before* the
  user commits to it, and `stage2_view.py:206-210` goes further and warns that
  the selected window is too short to test causality at all. Telling the user the
  graph will be empty *before* the run rather than after is exactly right.
- `stage2_view.py:295-301` resets the progress bar to 0 on failure, with a
  comment explaining that a bar parked at 55% next to failure text reads as
  "still working".
- `graph_panel.py:70-83` fills an untouched `QWebEngineView` so it does not paint
  blank white against the dark app.
- `data_view.py:109-131` blanks the counters on a read failure rather than
  leaving stale values, with a comment naming the "frozen numbers" symptom.

**Where it fails:**

**A successful "nothing found" is reported as a failure.**
`workers.py:94-96` routes `"No anomalies were detected in this observed window."`
through the `failed` signal. `stage2_view.py:295-300` then prefixes it with
`"Failed: "` and zeroes the progress bar. Finding no anomalies in a quiet window
is a correct, informative, *successful* outcome. Presenting it as an error trains
the user to distrust the tool and to widen the window until it reports something —
which is the opposite of the project's stated posture.

**No empty state on the default results tab.** `stage2_view.py:81-91` — covered
in §2. Both figure tabs got placeholders; the tab the user actually lands on did
not.

**Dead controls that look live.** `stage2_view.py:303-318` — both Export buttons
return silently when `_last_payload is None`. They are never disabled, are
rendered in full accent fill, and are the most prominent thing on the screen
before a run. Clicking one does nothing at all, with no message.

**No cancel.** Training runs for minutes (`workers.py:26-38`), RCA and incident
detection for tens of seconds (`workers.py:55-62`, `:86-104`). None of the three
`QThread`s exposes a cancel path, and closing the window during a run leaves a
running `QThread` with a live parent being destroyed.

**Progress granularity is coarse and unlabelled.** `stage2_view.py:76-79` places
a `QProgressBar` and an empty `QLabel` that are always present and always
occupying 46px, showing an empty trough at 1.11:1 contrast when idle
(`theme.py:149`) — in `docs/screenshots/stage2-rca-inference.png` it reads as a
third disabled input, not as a progress indicator. The incident scan
(`stage2_view.py:213-219`) reports no progress at all beyond a status string, and
it can take a while — it scores the model across up to a week of samples
(`workers.py:42-46`).

**The 30-second timers cause silent surprises.** `stage1_view.py:92-94` and
`data_view.py:105-107` refresh on a `QTimer`. `stage1_view.py:163` can flip the
Train button from enabled to disabled underneath the user's cursor when the
readiness recalculates. Nothing announces why.

**Errors are undifferentiated.** `stage1_view.py:193-195` and
`stage2_view.py:295-300` both write `f"Failed: {message}"` into a plain
`QLabel`. `theme.py:222` defines `QLabel#errorText { color: DANGER }` — and
nothing in `src/desktop/` ever sets that object name. The rule is dead; all
errors render in normal body colour.

---

## 7. Honesty in presentation — 4 / 10

This is the area where the gap between what the code *knows* and what the screen
*shows* is widest, and it matters most.

### The nuance exists — in the wrong widget

`src/reporting/report_generator.py:42-55` does exactly the right thing. Without a
surviving causal edge it refuses the phrase "root cause", says "**No causal
evidence.**", and when the top two candidates are within 0.01 it states outright
that "**Ranking is not meaningful here** ... their order is arbitrary."

That text goes into `self.report_text` — `stage2_view.py:122-124`, a read-only
`QPlainTextEdit`, which is the **fourth** tab, rendered as **raw unparsed
markdown** (the user sees literal `**No causal evidence.**` with asterisks), in
`TEXT_MUTED` at **6.15:1** (`theme.py:186-194`), in 10pt mono — the smallest,
lowest-contrast, most-deliberately-de-emphasised text style in the entire
application.

The default tab is "Root Causes". So the user's first and usually only view of
the result is a ranked table of four-decimal scores, and the sentence explaining
that the ranking is arbitrary is three tabs away in grey monospace with visible
asterisks. **The app is honest in its report and confident in its interface.**

### The table asserts precision the model does not have

`stage2_view.py:275` formats the composite as `f"{rc['composite_score']:.4f}"`.
Four decimal places claims resolution of 1 part in 10,000 on a number that is a
hand-weighted blend plus 0.30 × PageRank (`causal_inference/causal_engine.py:427-429`).
`report_generator.py:35` establishes that the project's own threshold for a
meaningful difference is **0.01** — two decimal places. The table renders four.
`0.8123` above `0.8119` reads as a real ordering; by the project's own stated
standard it is noise.

The `Rank` column (`stage2_view.py:275`) compounds this: it prints 1, 2, 3 as
integers with no indication that ranks 1 and 2 may be indistinguishable, and it
does so even when `confidence` is `"Correlation only"` for every row — i.e. when
`causal_engine.py:461-465` has explicitly determined that *nothing here was shown
to cause anything*.

### "Correlation only" is one uncoloured word in a 6-column table

`causal_engine.py:460-465` is a strong, deliberate piece of engineering: no
matter how high the score, without a surviving edge the label is "Correlation
only". In the UI that verdict is a 15-character string in the fourth column, at
the same weight, size and colour as the metric name and the score
(`stage2_view.py:279`). It has to compete with a bold "Rank 1" two columns to its
left and a four-decimal score one column to its left.

Read the row as a user would: **`1 | mem_pct | 0.8123 | Correlation only | 0.000 | —`**.
Three of those six cells assert confidence and one retracts it, and the retraction
is the least emphasised.

The `Outflow` column is the tell that is already on screen and going unused: when
no edge survives, `causal_outflow` is `0.000` for **every** row
(`stage2_view.py:276`), and `Downstream` is `—` for every row
(`stage2_view.py:277`). A full column of zeros and a full column of em-dashes is
the strongest possible visual evidence that nothing was proven — and it is
rendered in the same grey as everything else, with no annotation, so it reads as
missing data rather than as a finding.

### The empty causal graph loses its explanation and its theme

`visualizations.py:16-17` — when the graph has no nodes, it returns
`go.Figure().update_layout(title="No causal edges identified")` with **no
`paper_bgcolor`**. Every other path in that file sets `#151a2e`. So the single
most important honest state in the entire product — "we could not establish
causality" — renders as a **white rectangle** in a dark application with a small
default-styled title, looking like a broken chart rather than a deliberate
finding.

Worse, the careful explanation written for exactly this case
(`stage2_view.py:98-102`: "**No arrows means no causal link was established** —
which can be because none exists, or because the window was too short to test
one") is attached to the `PlotlyWebView` legend, which is unstyled
(`graph_panel.py:52` sets `figureLegend`; `theme.py` has no such rule) and sits
*above* the white box in body text. The distinction between "no causal link
exists" and "your window was too short to test for one" is the most
epistemically important sentence in the app and it has no visual treatment at
all.

### Where the design does get it right

- `stage2_view.py:206-210` — warning that the window is too short *before* the
  run.
- `stage2_view.py:164-170` — model drift is surfaced as a caveat and explicitly
  does **not** lock the stage, with a comment explaining that locking would be a
  wrong claim about the model rather than the window.
- `stage2_view.py:287-290` — the completion status names the raw counts
  ("N causal edge(s) survived correction, M process(es) attributed") rather than
  a percentage. This is the single most honest sentence rendered in the UI, and
  it lives in a plain `QLabel` above the results.
- `data_view.py:137-146` — reporting coverage as "2,524 of 4,487 expected (56%)"
  plus "6 breaks, 16.4 h not collected" rather than a span, with a comment
  explaining that span alone implies continuity it does not have. Exemplary.
- `consent.py:14-47` — the disclosure is specific, enumerates what is *not*
  recorded, and names the file path.

The pattern is clear: **the prose in this project is scrupulous and the visual
encoding is not.** Everywhere a caveat has been written as a sentence it is
excellent; everywhere meaning has to be carried by size, weight, colour or
position, the confident reading wins.

---

## Prioritised fixes

### (a) Quick wins — each under ~20 lines

Ordered by value. Nothing here changes what the pipeline computes.

1. **Cut the score to 2dp and mark statistical ties.** `stage2_view.py:275` —
   `f"{rc['composite_score']:.3f}"`, and when
   `abs(top - runner_up) < 0.01`, append `" (tied)"` to both rank cells or set
   both to `"1="`. *Cost:* the exact composite is no longer readable from the
   table; it remains in the JSON export (`stage2_view.py:311-318`), so nothing is
   lost for anyone who needs it.

2. **Style the `Correlation only` row so it does not read as a ranked result.**
   In `_on_finished` (`stage2_view.py:274-279`), when
   `rc["confidence"] == "Correlation only"`, set the Confidence item's foreground
   to `WARNING #f59e0b` (7.54:1 on `SURFACE_ALT`, AAA) *and* prefix the Rank cell
   with `"~"`. Colour alone would be a 1.4.1 failure; the `~` carries it for
   colourblind and screen-reader users. *Cost:* none.

3. **Give the Root Causes tab an empty state.** `stage2_view.py:81-91` — set
   `setRowCount(1)`, span the row, and put "Run an analysis to see ranked
   candidates." in it; on zero results after a run, "No candidates ranked — no
   metric crossed the anomaly threshold in this window." *Cost:* none; this is
   the fix for the known lower-area void.

4. **Fix the empty causal graph.** `visualizations.py:17` — add
   `paper_bgcolor="#151a2e", plot_bgcolor="#151a2e", font=dict(color="#e2e8f0"), height=520`
   to the early return, and change the title to "No causal edge survived
   correction — nothing here was shown to cause anything." *Cost:* none. This
   removes a white flash that currently reads as a bug.

5. **Add a focus ring to every interactive control.** `theme.py` — one block:
   `QPushButton:focus, QTabBar::tab:focus, QCheckBox:focus, QSlider:focus, QTableWidget:focus { border: 2px solid #2dd4bf; }`
   plus `outline: none` cleanup. *Cost:* ~6 lines. Fixes an outright WCAG 2.4.7
   failure across the whole app.

6. **Stop the blanket `QWidget` background painting boxes behind labels.**
   `theme.py:30-35` — change the selector to `QMainWindow, QDialog { ... }` and
   add `QLabel { background: transparent; }`. *Cost:* a handful of container
   widgets may need explicit backgrounds; verify against the two form-heavy tabs.
   This single change removes the striped artifact visible in both screenshots.

7. **Raise structural contrast to 3:1.** `theme.py:17` — `BORDER` from
   `rgba(240,246,252,0.10)` to `rgba(240,246,252,0.28)` (≈3.0:1 on `BG`);
   `theme.py:163` gridline to `0.16`; `theme.py:162`
   `alternate-background-color` to `#20262e`. *Cost:* the UI reads slightly
   busier. It is the difference between controls that look like controls and
   controls that look like bands.

8. **Fix the 150%-scaling position bug.** `main_window.py:45-47` — clamp before
   moving: `width = max(MINIMUM_SIZE[0], min(...)); height = max(MINIMUM_SIZE[1], min(...))`.
   *Cost:* three characters per line. Removes the 10px mis-centring at
   1920×1080 @150%.

9. **Report "no anomalies" as a result, not a failure.** `workers.py:94-96` — add
   an `empty` signal, or emit `finished_ok` with a flag; in `stage2_view.py`
   render it as a neutral status, not `"Failed: "`. *Cost:* ~8 lines across two
   files. Directly serves the honesty goal.

10. **Disable the Export buttons until there is something to export.**
    `stage2_view.py:131-134` — `setEnabled(False)` at construction, `True` at the
    end of `_on_finished`. *Cost:* none; it removes two dead controls that are
    currently the loudest things on screen.

11. **Give the two orphaned form rows accessible names.** `stage1_view.py:26` and
    `stage2_view.py:40` — `spin.setAccessibleName(label)` and
    `self.incident_combo.setAccessibleName("Detected incident")`. *Cost:* 3 lines.

12. **Style the figure legend so the caveat reads as a caveat.** `theme.py` — add
    `QLabel#figureLegend { color: #8b949e; font-size: 9.5pt; padding: 2px 4px; border-left: 2px solid rgba(240,246,252,0.20); }`
    and a `#figureTitle` rule. *Cost:* none; three object names already exist in
    `graph_panel.py` waiting for rules.

13. **Make the inline figures fill their panel.** `graph_panel.py:89` — pass
    `fill=True` from `show_figure`. The code path already exists
    (`graph_panel.py:101-113`). *Cost:* the fixed 420/520 heights stop applying
    inline, which is the point; verify the timeline still reads at ~150px.

14. **Reduce Stage 1's console dominance.** `stage1_view.py:87` — drop
    `stretch=1` to `setMaximumHeight(200)` and give the stretch to a spacer, or
    put the console in a collapsible group box. *Cost:* long training logs need
    scrolling, which they already do.

15. **Fix the Captured Data column gulf.** `data_view.py:95` and `:178` — stretch
    column 2 ("Latest value") instead of column 1, or set column 1 to
    `ResizeToContents` and add a trailing stretch column. *Cost:* none.

16. **Right-align and monospace the numeric columns.**
    `stage2_view.py:278-279` — `setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)`
    and a `QFont("Cascadia Mono")` on the Score and Outflow columns. *Cost:*
    none; makes a column of scores scannable as a column.

17. **Remove the dead line and the dead rule.** `stage2_view.py:88`
    (`setStretchLastSection` is inert under `Stretch` mode) and `theme.py:222`
    (`#errorText` is never applied — either wire it into
    `stage1_view.py:194` / `stage2_view.py:300` or delete it).

18. **Add tab shortcuts.** `main_window.py:87-89` — three `QShortcut`s for
    `Ctrl+1/2/3`. *Cost:* ~4 lines.

### (b) Larger redesigns

**B1. A verdict banner above the results tabs.** *This is the single
highest-value change in the review.*

Between `stage2_view.py:79` and `:80`, insert a result header that states the
finding in one sentence at 16–18pt, driven by the same `has_causal_evidence`
branch that `report_generator.py:33` already computes:

- With causal evidence: `mem_pct` at 18pt bold, `ACCENT`, with
  "3 causal edges survived FDR correction" beneath at 9.5pt muted.
- Without: `mem_pct` at 18pt in `TEXT` (not accent — it is not a claim), with
  "**Correlation only.** No causal edge survived. This is the metric that
  deviated earliest and hardest, not a demonstrated cause." beneath, in
  `WARNING`, and where `too_close` holds, "`swap_pct` scores within 0.01 — their
  order is arbitrary."

This gives the app the thing it currently lacks entirely: one place where the
answer is the largest text on screen, and where the *strength* of the answer is
encoded in the same glance. It fixes hierarchy (§1), fills the top of the results
area (§2), and moves the honesty from the fourth tab to the first thing read
(§7).

*Cost:* ~60 lines in `stage2_view.py` plus two `theme.py` rules. Requires
`_on_finished` to receive `evidence["surviving_causal_edges"]` (already in the
payload, `stage2_view.py:286`) and to compute `too_close` from the top two
composites (trivial). Adds ~90px of permanent vertical chrome to the results
area, which argues for doing B2 at the same time. No pipeline change.

**B2. Collapse the configuration panel after a successful run.**

`stage2_view.py:29-64` occupies the top ~260px permanently. Make the group box
checkable/collapsible and collapse it on `_on_finished`, showing a one-line
summary of what was analysed ("2026-07-28 23:32 → 2026-07-29 23:32 · lag 5 ·
2,880 samples") that expands on click. *Cost:* ~40 lines. Users who re-run with
tweaked parameters take one extra click; in exchange the results area roughly
doubles, and the 1024×640 minimum becomes genuinely usable. Pairs with B1 —
together they roughly triple the vertical space given to the answer.

**B3. Render the Report tab as rich text.**

`stage2_view.py:122-124` — replace the `QPlainTextEdit` with a read-only
`QTextBrowser` and `setMarkdown(...)`. The bold caveats
(`report_generator.py:46-48`, `:51-55`) would then actually render bold instead
of showing literal asterisks, and the tab could inherit `TEXT` rather than
`TEXT_MUTED`. *Cost:* ~15 lines plus a `theme.py` block for `QTextBrowser`;
loses the monospace alignment of any tabular content in the report, so check the
process-attribution section renders acceptably. Copy-to-clipboard behaviour
changes from raw markdown to rich text — if the raw markdown matters, keep the
Export button as the path for that (it already is).

**B4. Rebuild the figures on the app palette, with non-colour encoding.**

`visualizations.py` end to end: import the palette from `theme.py` rather than
hardcoding `#151a2e`/`#667eea`/`#ff4757`; replace the red/orange node pair with
shape *and* size variation (diamond for root cause, circle for source) so role
survives colourblindness; drop the emoji from the legend
(`visualizations.py:65`); add dash patterns to the timeline series
(`visualizations.py:108`) so five metrics are separable without colour; raise the
lag annotation from `size=9` to `size=11`. *Cost:* ~50 lines and a visual
regression pass over both figures. `graph_panel.py:80` and `:117` would import
the background from `theme.py` too, killing the last hardcoded duplicates. This
is the fix for the two-colour-systems problem in §3 and most of the colour-alone
problem in §4.

**B5. Cancellable long operations.**

`workers.py` — add a cooperative cancel flag checked in the `progress` callback,
and swap the Run/Train buttons to "Cancel" while a worker is live. *Cost:* ~40
lines across `workers.py`, `pipeline/engine.py` (the progress callback would need
to be able to raise or return a stop signal) and the two views. Highest cost of
anything here because it touches the pipeline; justified by a training run that
can take minutes with no exit.

**B6. Wrap each tab's content in a `QScrollArea`.**

No view has one. *Cost:* ~5 lines per view plus care that the stretching widget
(`results_tabs`, `log_console`, the channel table) is not itself inside the
scrolled region, or the scrolling nests badly. This is what makes the declared
1024×640 minimum and 150%+ scaling genuinely safe rather than nearly-safe;
without it, `data_view.py:94`'s 430px minimum has nowhere to go.
