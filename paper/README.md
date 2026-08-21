# IEEE conference paper

`localrca_ieee.tex` — the project written up in IEEE conference format
(`IEEEtran`, two-column).

## Status, stated plainly

**This has never been compiled.** No TeX distribution is installed on the
development machine, so the source has been checked structurally
(`python check_tex.py`) but not run through `pdflatex`. Expect to fix small
things on first build — that is normal for untested LaTeX, and pretending
otherwise would be the same kind of unverified claim the paper itself argues
against.

`check_tex.py` verifies what can be verified without TeX: balanced
environments and braces, every `\ref` has a `\label`, every `\cite` has a
`\bibitem`.

## Building

Needs a TeX distribution — [MiKTeX](https://miktex.org/) on Windows, or TeX
Live. `IEEEtran`, `tikz` and `pgfplots` are included in both; MiKTeX will
offer to fetch anything missing on first run.

```bash
pdflatex localrca_ieee
pdflatex localrca_ieee     # twice, so cross-references resolve
```

No `bibtex` step: the bibliography is a `thebibliography` block inside the
document. If you move to a `.bib` file, add `bibtex localrca_ieee` between the
two `pdflatex` runs.

The easiest zero-install route is [Overleaf](https://overleaf.com) — upload
the `.tex` alone and it compiles, since there are no external assets.

## Figures

All four are drawn in TikZ/pgfplots rather than imported, so the document is
self-contained and the figures restyle with the document font.

| Figure | Content |
|---|---|
| 1 | Process architecture — collector, database, desktop app, supervisor |
| 2 | Inference pipeline, showing all four terminal states |
| 3 | Causal yield against window length (bar chart) |
| 4 | Subsystem prior as a total order, with the rejected directions |

**Every number in the figures is measured**, not illustrative. Figure 3 comes
from `outputs/causal_yield_lag5.json`, produced by
`tools/measure_causal_yield.py`; Figure 4's rejection counts come from
`tools/audit_topology_map.py`. Re-run either tool and the figures can be
updated from real output.

## Relationship to the other documents

- `docs/IMPLEMENTATION_PAPER.md` — the long-form version, ~1,800 lines. Every
  measurement, every defect, and the reasoning behind each design decision.
  The LaTeX paper is a condensation of it for a conference page limit.
- `docs/PRODUCTION_CHECKLIST.md` — what still stands between the project and
  something shippable to strangers.

Where the two disagree, the markdown is authoritative: it is updated as
measurements are taken, and the LaTeX is a snapshot.

## Before submitting

- Fill in the author block — institution and location are placeholders.
- Check the page limit. The current draft is written for a 6–8 page venue;
  Section VI (topology audit) is the most self-contained if you need to cut.
- Re-run `tools/measure_causal_yield.py` if the database has grown, and update
  Figure 3, Table I and the abstract's numbers together. They are quoted in
  several places and will drift apart if updated individually.
