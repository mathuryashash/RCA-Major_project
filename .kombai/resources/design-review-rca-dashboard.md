# Design Review Results: AI-Powered RCA Dashboard

**Review Date**: 2026-03-10
**Page**: `src/reporting/dashboard.py` (Streamlit single-page app)
**Focus Areas**: Visual Design · UX/Usability · Responsive/Mobile · Accessibility · Performance · Charts/Graphs

> **Note**: This review was conducted through static code analysis only. Visual inspection via browser would provide additional insights into layout rendering, interactive behaviors, and actual appearance.

---

## Summary

The dashboard has a strong visual identity with a cohesive dark/purple theme and good use of glassmorphism cards. However, the CSS design system relies entirely on hardcoded values with no CSS variables, accessibility features are largely absent from the custom HTML blocks, and the two-stage pipeline UX creates a confusing flow for first-time users. Chart labeling and causal graph clarity are the most impactful issues for the core use case.

---

## Issues

| # | Issue | Criticality | Category | Location |
|---|-------|-------------|----------|----------|
| 1 | Gradient text (`-webkit-text-fill-color: transparent`) renders as **invisible in Windows High Contrast Mode** — hero title, KPI values, sidebar heading all affected | 🔴 Critical | Accessibility | `dashboard.py:79-82, 137-143, 559-561` |
| 2 | **No `@media (prefers-reduced-motion)`** override — `titleGlow`, `fadeIn`, `arrowPulse` animations run unconditionally, violating WCAG 2.1 SC 2.3.3 | 🔴 Critical | Accessibility | `dashboard.py:89-93, 218-221, 389-393` |
| 3 | **Stage 2 has no guard** — users can switch to Stage 2 and click "Run RCA Inference" without ever running Stage 1 / training the model, causing a silent failure or crash | 🔴 Critical | UX/Usability | `dashboard.py:1041-1100` |
| 4 | **All colors are hardcoded** — `#667eea`, `#764ba2`, `#f093fb`, `#8892b0`, `#ff4757` etc. appear 30+ times across the 416-line CSS block with zero CSS custom properties. Any theme change requires hunting every occurrence | 🔴 Critical | Visual Design | `dashboard.py:62-478` |
| 5 | **Low contrast text** — `.footer-text` uses `color: #4a5568`, `.about-desc` uses `color: #718096`, `.kpi-label` uses `color: #8892b0`, `.chip-name` uses `color: #8892b0` all on dark backgrounds. Estimated contrast ratios below 4.5:1 (WCAG AA) | 🟠 High | Accessibility | `dashboard.py:148-151, 299-303, 308-312, 424-427` |
| 6 | **Confidence gauge color is the only status signal** — gauge changes from green → orange → red with no accompanying text label ("Low / Medium / High") or icon. Fails WCAG 1.4.1 (Use of Color) | 🟠 High | Accessibility | `dashboard.py:1288-1299` |
| 7 | **`.about-grid` fixed at `repeat(3, 1fr)`** without responsive breakpoints — overflows on screens < ~600px | 🟠 High | Responsive | `dashboard.py:268-270` |
| 8 | **`.metric-grid` fixed at `repeat(5, 1fr)`** without responsive breakpoints — metric chips will be tiny or overflow on tablets/mobile | 🟠 High | Responsive | `dashboard.py:396-398` |
| 9 | **Stage 1 & Stage 2 KPI grids** use inline `grid-template-columns: repeat(4, 1fr)` with no media query — overflow on narrow screens | 🟠 High | Responsive | `dashboard.py:901, 1048, 1266` |
| 10 | **Sidebar clutter** — 8 controls (pipeline stage, baseline days, epochs, window size, seed, failure type, severity, Granger lag) with no logical visual grouping. Users must scroll to find relevant controls per stage | 🟠 High | UX/Usability | `dashboard.py:565-615` |
| 11 | **No persistent "Model Ready" state** — After training in Stage 1 the success message disappears on any interaction; when users switch to Stage 2 there is zero confirmation the model exists. The `st.cache_resource` state is invisible to the user | 🟠 High | UX/Usability | `dashboard.py:987-1034` |
| 12 | **"Run RCA" button buried below two large cards** — The action button (line 1100) appears only after scrolling past the configuration summary card and the scenario details card. Primary CTA should be above or alongside the supporting context | 🟠 High | UX/Usability | `dashboard.py:1043-1100` |
| 13 | **Causal graph: timeline annotation collision** — All `add_vline` annotations use `annotation_position="top left"`. When multiple metrics become anomalous within close time windows the labels overlap and are unreadable | 🟠 High | Charts | `dashboard.py:1396-1404` |
| 14 | **Bar chart x-axis labels truncated** — `px.bar(score_df, y="Max Score")` with metric names like `memory_usage_percent` as index — Plotly defaults will angle/clip them without explicit `tickangle` or `tickfont` settings | 🟠 High | Charts | `dashboard.py:1422-1430` |
| 15 | **Weak cache key `_df_hash`** — hash is computed as `hash(df.shape) ^ hash(tuple(df.columns.tolist()))`. Two DataFrames with the same shape and column names but different content hash identically — stale cached model can be silently returned | 🟠 High | Performance | `dashboard.py:642-644, 630-639` |
| 16 | **Custom HTML elements have no ARIA roles** — `.kpi-card`, `.glass-card`, `.timeline-item`, `.stage-badge`, `.confidence-gauge` are all plain `<div>` blocks rendered via `unsafe_allow_html=True`. Screen readers have no semantic context | 🟡 Medium | Accessibility | `dashboard.py:900-919, 1044-1067, 1286-1299` |
| 17 | **Pipeline flow arrows orphan on wrap** — `.pipeline-flow` uses `flex-wrap: wrap` with `gap: 0` and plain `→` span arrows in HTML. When steps wrap to a new line the arrow characters detach from their source step | 🟡 Medium | Responsive | `dashboard.py:345-346, 926-957` |
| 18 | **KPI card values overridden with inline color/size** in Stage 2 — breaks the `.kpi-value` design system class and creates visual inconsistency vs Stage 1 KPI cards | 🟡 Medium | Visual Design | `dashboard.py:1050-1063` |
| 19 | **"~35K Model Parameters" is hardcoded** in Stage 1 KPI card — the actual parameter count changes with `window_size` and `n_features` but the card always shows `~35K` | 🟡 Medium | UX/Usability | `dashboard.py:912` |
| 20 | **Benchmark runs 6 scenarios synchronously** in the UI thread — blocks the entire Streamlit server for all concurrent users during benchmark; no parallelism or background execution | 🟡 Medium | Performance | `dashboard.py:1506-1533` |
| 21 | **416-line CSS block re-injected on every render** — The entire `<style>` block is passed to `st.markdown()` at module level, so Streamlit re-injects it on every widget interaction. Should be loaded from an external file or placed in a `_st_static` approach | 🟡 Medium | Performance | `dashboard.py:62-478` |
| 22 | **Causal graph edge strength not visually encoded** — `strength` attribute is computed and stored on edges but not used to vary edge `width` or `color` in the Plotly figure. Only `lag` labels are shown, wasting a key data dimension | 🟡 Medium | Charts | `dashboard.py:663-696` |
| 23 | **Causal graph: node label overlap** — all nodes use `textposition="top center"` with no collision avoidance. Dense graphs will have unreadable overlapping labels | 🟡 Medium | Charts | `dashboard.py:728-730` |
| 24 | **"About This System" expander is collapsed by default** (`expanded=False`) — first-time users see no onboarding context. For a complex two-stage ML pipeline, the getting-started info should be prominent on first visit | 🟡 Medium | UX/Usability | `dashboard.py:500` |
| 25 | **"Accuracy Benchmark" tab is a secondary workflow** but appears as the 6th tab alongside core results tabs (Root Causes, Graph, Timeline). It requires a separate button click inside the tab, making it feel like a hidden feature | 🟡 Medium | UX/Usability | `dashboard.py:1304-1311` |
| 26 | **Emojis in button labels** (`🚀 Generate Data & Train Model`, `▶ Simulate Incident & Run Full RCA`) are read verbatim by screen readers ("rocket Generate Data…", "black right-pointing triangle Simulate…") | 🟡 Medium | Accessibility | `dashboard.py:985, 1100` |
| 27 | **`incident_scaled.set_index("timestamp")` computed multiple times** — called in the Granger causality block, in the timeline tab, and implicitly in the benchmark — same DataFrame transformation repeated without memoization | ⚪ Low | Performance | `dashboard.py:1203, 1389` |
| 28 | **Confidence gauge CSS transition never plays** — `transition: width 1s ease` on `.gauge-bar` is defined (line 181) but Streamlit re-renders the entire component from scratch on each run, so there is no previous DOM state to animate from | ⚪ Low | Visual Design | `dashboard.py:178-183` |
| 29 | **`feat_cols_preview` recomputes on every render** in Stage 1 — `[c for c in normal_df.columns if c != "timestamp"]` is recalculated independently of `feat_cols` from the cached model, creating a potential mismatch | ⚪ Low | Performance | `dashboard.py:997` |
| 30 | **Root cause table "Downstream" column silently truncates** — `[:3]` limits shown downstream effects with no indication more exist (no count badge, no tooltip, no "…") | ⚪ Low | UX/Usability | `dashboard.py:1328` |

---

## Criticality Legend

- 🔴 **Critical**: Breaks functionality or violates accessibility standards (WCAG)
- 🟠 **High**: Significantly impacts user experience, data accuracy, or design quality
- 🟡 **Medium**: Noticeable issue that should be addressed in a regular sprint
- ⚪ **Low**: Nice-to-have improvement or minor polish

---

## Next Steps (Recommended Priority Order)

### Immediate (🔴 Critical)
1. **Add CSS custom properties** — Extract all color/spacing values into `:root` variables at the top of the CSS block. Eliminates the hardcoded-value sprawl in one pass.
2. **Add `prefers-reduced-motion` media query** — Wrap all `@keyframes` animations with `@media (prefers-reduced-motion: no-preference)`.
3. **Guard Stage 2** — Check `st.session_state` for a trained model flag before rendering Stage 2 content. Show a clear call-to-action to complete Stage 1 first if missing.
4. **Fix gradient text High Contrast fallback** — Add a `@media (forced-colors: active)` block that sets `color: ButtonText` and removes `-webkit-text-fill-color: transparent`.

### High Priority (🟠)
5. **Responsive grids** — Add `min(100%, Xpx)` or `auto-fill/auto-fit` variants for `.about-grid`, `.metric-grid`, and the inline KPI grids. Break to 2-col on < 768px and 1-col on < 480px.
6. **Sidebar: Separate controls by stage** — Show only stage-relevant controls based on the selected pipeline stage radio. Stage 1 controls: baseline days, epochs, window size, seed. Stage 2 controls: failure type, severity, Granger lag.
7. **Add "Model Trained ✅" indicator** — Use `st.session_state` to persist a training status flag shown persistently in the sidebar or at the top of Stage 2.
8. **Fix chart annotations** — Alternate `annotation_position` values (top left, top right, bottom left, bottom right) based on index to avoid collision. Add `tickangle=-45` to bar chart x-axis.
9. **Strengthen cache key** — Use `hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()` instead of the shape-based `_df_hash`.
10. **Add confidence level text label** — Alongside the color-coded gauge, add a text label: `"Low Confidence"` / `"Medium Confidence"` / `"High Confidence"`.

### Medium Priority (🟡)
11. **Encode edge strength visually** — Map `strength` to Plotly edge `line.width` (e.g., `width = 1 + strength * 5`) and use a color gradient from light to `#667eea`.
12. **Move "About" expander to `expanded=True` on first visit** — Use `st.session_state` to track first render.
13. **Move Benchmark to a separate sidebar button / standalone expander** — Decouple from the 6-tab results view so it can be accessed independently.
14. **Replace emoji in `st.button()` labels** with text only, or use `aria-hidden` workaround via custom HTML.
15. **Extract CSS to `src/reporting/style.css`** — Load once with `st.markdown(f"<style>{open('style.css').read()}</style>", unsafe_allow_html=True)` to reduce re-injection overhead.
