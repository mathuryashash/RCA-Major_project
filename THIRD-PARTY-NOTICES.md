# Third-Party Notices

LocalRCA is distributed as a frozen binary that embeds the libraries below.
Their licences apply to those components, not to LocalRCA's own source, which
is MIT (see `LICENSE`).

## PySide6 / Qt — LGPL v3

This is the obligation that actually constrains distribution, and it is worth
stating plainly rather than burying.

PySide6 and the Qt libraries it wraps are used here under the **LGPL v3**. The
binary in `dist/RCA-Desktop/_internal` contains Qt shared libraries
(`Qt6Core.dll`, `Qt6Gui.dll`, `Qt6Widgets.dll`, `Qt6WebEngineCore.dll` and
others). LGPL permits this, provided the recipient can replace those libraries
with their own build:

- The Qt DLLs are shipped as **separate, unmodified shared libraries** in
  `_internal`, not statically linked into the executable. A recipient may
  replace them with a compatible build of their own.
- Neither Qt nor PySide6 has been modified.
- Qt sources: <https://download.qt.io/official_releases/qt/>
  PySide6 sources: <https://download.qt.io/official_releases/QtForPython/>
- LGPL v3 text: <https://www.gnu.org/licenses/lgpl-3.0.html>

If the packaging is ever changed to link Qt statically or to obfuscate the
`_internal` directory, this permission no longer holds and the distribution
would need re-examining.

## Other bundled components

| Component | Licence | Use here |
|---|---|---|
| PyTorch | BSD-3-Clause | LSTM autoencoder training and inference |
| NumPy | BSD-3-Clause | numerical arrays |
| pandas | BSD-3-Clause | telemetry frames |
| scikit-learn | BSD-3-Clause | scaling, ensemble detection |
| statsmodels | BSD-3-Clause | Granger causality, ADF stationarity |
| NetworkX | BSD-3-Clause | causal graph construction and PageRank |
| Plotly | MIT | causal graph and timeline figures |
| psutil | BSD-3-Clause | system and process sampling |
| pywin32 | PSF-style | Windows Event Log access |
| optree | Apache-2.0 | PyTorch pytree support (bundled; see packaging notes) |
| SQLite | Public domain | local telemetry store |

Each of these permits redistribution in binary form with attribution. Their
full licence texts ship inside the respective package directories under
`_internal`.

## Reproducing this list

`requirements.txt` pins the direct dependencies. The bundled set is larger
because PyInstaller collects transitive dependencies; `packaging/excludes.txt`
records what is deliberately left out and why.
