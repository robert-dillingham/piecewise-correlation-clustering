# Project Plan: Piecewise Correlation (Python Package)

*Living document. Update the [Update Log](#13-update-log) and [Decision Log](#12-decision-log) whenever scope changes.*

**Status (2026-11-18):** Phase 0 (repository setup) is complete. Phase 1 (data model and class skeleton) is next.

---

## 1. Overview

**Piecewise correlation** extracts information from the *relationships between* grouped time series rather than from each series in isolation. When many groups (stocks, sensors, regions, etc.) are measured at the same timestamps under the same conditions, the shared environment leaves a fingerprint on all of them. Comparing series to each other, summarized over several time scales, surfaces structure that single-series methods miss.

The method in one paragraph:

1. Standardize each group's series so they are comparable.
2. Summarize each series at several time scales (hourly, daily, weekly, monthly, or custom).
3. Compute a correlation (r²) matrix between all groups at each scale.
4. Combine the matrices into a single weighted-average matrix, which is the similarity measure used downstream.

This project ports and extends an earlier R implementation into a clean, tested, documented Python package, primarily as a GitHub portfolio piece.

## 2. Goals

| # | Goal | Description |
|---|------|-------------|
| 1 | **Clustering** | Use the combined matrix to group time series that contain only values and timestamps. The R version did this faster and more accurately than traditional methods. We need to reproduce and *demonstrate* that claim in Python. |
| 2 | **Prediction** | Match the current window of one group against equivalent past periods (this month vs. last month, this year vs. last year) of all groups, find the closest match, and use that match's subsequent movement to extrapolate the target series. |

**Non-goals (for now):** being a general-purpose data downloader, real-time/streaming support, or trading advice. Data fetching (e.g., Alpaca) lives in `examples/` only, never as a core dependency. Downloaded market data is not committed to the repository (see D8).

## 3. Current State

The original prototype (`general-testing.py`, now `notebooks/testing-notebook.ipynb`, excluded from linting) demonstrated the full calculation on 10 S&P 100 tickers with hourly bars from Alpaca. The flow is: fetch data → convert to New York time → scale each series by its maximum → aggregate to four periods with a weights dict → interpolate → pairwise r² matrix per period → weighted sum into `corr_dict['result']`.

Two example scripts now exist:

- `examples/download_sp100.py` downloads hourly IEX bars for the S&P 100 (2025-01-01 onward, fixed end date) and writes `examples/data/sp100_hourly.parquet` in tidy long format (`symbol`, `timestamp`, OHLCV, ...). Credentials come from a local `.env` file.
- `examples/stocks_alpaca.py` reads that Parquet file and reproduces the prototype's calculation with a wide frame, one vectorized `corr()` per period, and explicit weight lookup. It intentionally keeps the full-sample max scaling and the interpolation behavior so the Phase 1 parity test has a baseline.

### Issues to fix during the port

| Issue | Why it matters | Planned fix | Status |
|-------|----------------|-------------|--------|
| **API key and secret were hardcoded in the script** | Anyone with repo access can use them; they must never be committed | Rotate the keys. Load from `.env` via `python-dotenv`. Add `.env` to `.gitignore`. Provide `.env.example`. | **Done** (Phase 0): keys revoked and removed, `.env` ignored, `.env.example` committed |
| Missing `from io import StringIO` | Script fails on the `read_html` line | Fix in the example script. | **Done** (`download_sp100.py`) |
| Scaling uses the **full-sample max** (`x / max(x)`) | Leaks future information. Harmless for clustering, but invalid for prediction/backtesting. | Make scaling configurable (`max`, `zscore`, `first`, `none`) and add an *expanding* (no-lookahead) mode for prediction. | Open (Phase 1 option, Phase 7 expanding mode, D7) |
| Empty time bins are likely filled by interpolation | Time-based grouping creates bins for overnight/weekend periods (NaN means), and `interpolate()` then invents values there, which smooths the series and can inflate correlations. **Verify with a test**, then fix. | Drop empty bins by default. Interpolate only real gaps (limit and policy configurable). | Open (Phase 2); still present in `stocks_alpaca.py` |
| `corr() ** 2` discards sign | Anti-correlated groups look identical to correlated ones | Make the similarity transform configurable (`r2`, `abs`, `signed`). | Open (Phase 1) |
| Weighted sum uses `zip` over two dicts | Silently wrong if dict order ever differs | Look up weights explicitly by key. | Fixed in the example; class implementation in Phase 1 |
| O(n²) Python loop over `Series.corr` | Slow for hundreds of groups | Use one vectorized `DataFrame.corr()` on the wide frame per period. | Fixed in the example; class implementation in Phase 1 |
| `symbols[:10]` repeated and hardcoded | Easy to get inconsistent | Drive everything from the input data's groups. | Fixed in the example |
| Result is a loose dict | Not reusable | Replace with a class (see §5). | Open (Phase 1) |

## 4. Repository Layout

```
piecewise-correlation-clustering/
├── pyproject.toml
├── README.md
├── LICENSE
├── project_plan.md
├── .gitignore                  # includes .env, .venv*/, .ipynb_checkpoints/, examples/data/
├── .env.example                # template for download_sp100.py credentials (.env itself is ignored)
├── .github/
│   └── workflows/
│       └── ci.yml              # ruff check, ruff format --check, pytest on Python 3.10-3.12
├── src/
│   └── piecewise_correlation/  # package name: see Decision Log D1
│       ├── __init__.py
│       ├── data.py             # input validation → canonical tidy DataFrame
│       ├── missing.py          # NaN handling policies
│       ├── scaling.py          # standardization options
│       ├── periods.py          # built-in period registry + custom label handling
│       ├── aggregate.py        # summarization functions (mean, median, custom)
│       ├── model.py            # PiecewiseCorrelation class (fit, matrices, combine)
│       ├── cluster.py          # clustering methods
│       ├── predict.py          # prediction methods
│       └── plotting.py         # heatmap, dendrogram (optional dependency)
├── tests/
│   ├── fixtures/               # small committed fixtures (synthetic and/or R-implementation CSVs)
│   └── test_*.py
├── examples/
│   ├── download_sp100.py       # one-time download → examples/data/sp100_hourly.parquet
│   ├── stocks_alpaca.py        # cleaned prototype; reads the Parquet file
│   └── data/                   # gitignored; generated locally (see D8)
├── notebooks/                  # exploration only; logic lives in src/ (excluded from ruff)
│   └── testing-notebook.ipynb  # original prototype
└── docs/
```

**Dependencies:** core = `numpy`, `pandas>=2.2` (the `h` / `ME` / `QE` frequency aliases), `scipy` (hierarchical clustering). Optional extras: `plot` (matplotlib/seaborn), `examples` (alpaca-py, python-dotenv, requests, lxml, pyarrow), `dev` (pytest, pytest-cov, ruff, mypy, jupyterlab). Build backend is hatchling with `packages = ["src/piecewise_correlation"]`; `requires-python = ">=3.10"`. Install for development with `python -m pip install -e ".[dev,examples]"`.

## 5. Core Design

### 5.1 Canonical data model (Updates 1 and 7)

All input is converted internally to a **tidy long DataFrame**:

| group | timestamp | value | *(optional custom label columns...)* |
|-------|-----------|-------|------|
| AAPL | 2026-01-02 14:30 | 273.2 | 1 |

- One row per (group, timestamp). Timestamps must be datetime-like and shared across groups (same cadence).
- Extra columns are allowed and are what users reference for **custom periods** (§5.4).
- Adapters accept: a tidy DataFrame, a wide DataFrame (index = timestamp, columns = groups), and a MultiIndex Series/DataFrame like Alpaca's `(symbol, timestamp)`.
- Series objects are **not** a first-class input. They are converted to DataFrames up front.

Validation (clear errors, not silent failures): required columns exist, timestamps parse and are sorted, no duplicate (group, timestamp) pairs, at least 2 groups, requested periods are not finer than the data's cadence, timezone handling is explicit.

### 5.2 The model class (Update 7)

A scikit-learn-style estimator that is configured once, fit on data, and exposes results as attributes and methods.

```python
from piecewise_correlation import PiecewiseCorrelation

model = PiecewiseCorrelation(
    periods={"hourly": 4, "daily": 3, "weekly": 2, "monthly": 1},  # name → weight
    agg="mean",                  # or "median", "max", callable, ...
    scale="max",                 # standardization
    missing="time",              # NaN policy
    similarity="r2",             # "r2" | "abs" | "signed"
)
model.fit(df, group="symbol", time="timestamp", value="high")

model.matrices_["daily"]    # labeled DataFrame, .loc["AAPL", "MSFT"]
model.combined_             # weighted-average matrix
model.weights_              # normalized weights
model.cluster(n_clusters=5)         # → pandas Series: group → cluster label
model.predict("AAPL", horizon=...)  # → forecast (see §7)
```

Design rules:
- `fit` returns `self`. Fitted attributes end with `_`. No work happens in `__init__` beyond validating parameters.
- Results are labeled DataFrames, so `matrix.loc[row, col]` works and `.to_numpy()` is available for heavy computation.
- Weights are normalized internally and validated (positive, non-empty). Keys map to the same names used for the period matrices.
- Keep the object picklable and add `to_dict()` / `to_frame()` for exporting.

### 5.3 Missing data (Update 2)

`missing` accepts a policy string or a small `MissingPolicy` object.

- Methods: `"time"` (default), `"linear"`, `"ffill"`, `"bfill"`, `"drop"`, `"none"`.
- Options: `limit` (max consecutive fills), `min_coverage` (drop a group if too much of its data is missing), `fill_edges` (leading/trailing NaNs).
- **Only fill genuine gaps.** Structural gaps (overnight, weekends, holidays) must not be interpolated. Detect them by dropping empty bins rather than filling them.
- Record what was filled per group (`model.fill_report_`) so the user can see how much of each series is imputed.
- Document the lookahead caveat: interpolation uses future values, so it is for clustering/exploration. `ffill` is the safer choice for prediction.

### 5.4 Periods and custom labels (Updates 3 and 4)

Two kinds of periods behave differently, and the design must keep them distinct.

1. **Time-binned periods** (`hourly`, `daily`, `weekly`, ...): group by calendar bin. The result is a time-indexed series with one value per bin across the full history.
2. **Label-based periods** (custom, and cyclical built-ins like `month_of_year` or `weekday`): group by a label column. The result is indexed by the label (e.g., 1-12), which *collapses across years*. This answers "do these groups behave alike across the same phase of a cycle?" rather than "do they move together over time?"

Built-in registry (`periods.py`), mapping a friendly name to a pandas alias or label function:

| Name | Type | Definition |
|------|------|-----------|
| `seconds`, `minutes`, `hourly` | binned | `"s"`, `"min"`, `"h"` |
| `daily`, `weekly`, `monthly`, `quarterly`, `yearly` | binned | `"D"`, `"W"`, `"ME"`, `"QE"`, `"YE"` |
| `weekday` | label | day-of-week 0-6 |
| `hour_of_day`, `month_of_year` | label | cyclical labels |

**Custom periods:** the user adds a column of labels to their DataFrame (e.g., `1` for every January value, `2` for every February value) and references it by column name in `periods`:

```python
df["fiscal_quarter"] = ...
PiecewiseCorrelation(periods={"daily": 3, "fiscal_quarter": 2})
```

Resolution order for each key: custom column in the data → built-in registry → raise a clear error listing valid options. Aggregation is `groupby([group, label_or_bin])`.

### 5.5 Aggregation (Update 5)

- One `agg` setting applies to **all** periods.
- Accept string names (`"mean"`, `"median"`, `"max"`, `"min"`, `"sum"`, `"std"`, `"first"`, `"last"`) or any callable that maps a Series to a scalar (including lambdas).
- Default is `"mean"`. Validate that a callable returns a scalar.

### 5.6 Similarity and combining

- Correlation per period via one `DataFrame.corr(min_periods=...)` on the wide frame.
- Transform: `r2` (default, matches the prototype), `abs`, or `signed`.
- Combined matrix = Σ (normalized weight × period matrix). Handle NaN entries (pairs with too little overlap) explicitly: re-normalize weights over the periods that are available for that pair, or flag it. This is **Decision D4**.
- Sanity checks in tests: symmetric, diagonal = 1, values within [0, 1] for `r2`.

## 6. Clustering (Goal 1)

Convert similarity to distance (`1 - combined`) and cluster:

- **Default:** agglomerative/hierarchical clustering via SciPy (`linkage` + `fcluster`), with linkage selectable (`average`, `complete`, `ward` is not valid for precomputed distances without care).
- **Options to add later:** spectral clustering on the similarity matrix, HDBSCAN, k-medoids (optional dependencies).
- Outputs: a labeled `Series` of cluster assignments, plus the dendrogram linkage matrix for plotting.
- Selecting the number of clusters: explicit `n_clusters`, a distance `threshold`, or a helper that scores a range with silhouette on the precomputed distance.

**Evaluation (this is what proves the "faster and more accurate" claim):**
- Synthetic datasets with *planted* clusters (shared latent factors + noise) so ground truth is known; score with Adjusted Rand Index and NMI.
- Baselines: plain Pearson on raw series, Euclidean k-means, and DTW-based clustering (optional `tslearn`).
- Real-data demo: S&P 100 against GICS sector labels as a loose reference.
- Benchmarks for speed as number of groups and series length grow.

## 7. Prediction (Goal 2)

This is the least-defined part of the project, so it is described as a hypothesis to be refined through experiments.

**Procedure (analog / nearest-neighbor forecasting):**

1. Choose a target group and the current window (e.g., the month so far).
2. Build candidates: every group's data over the *equivalent* windows in prior periods (same position within last month, last year, etc.), aligned by position.
3. Score each candidate against the target's current window with the piecewise similarity.
4. Take the best match (or the top-k, weighted by similarity).
5. Extrapolate: apply the candidate's *subsequent* relative movement to the target's last observed value.

**Open questions (to settle through experiments):**
- Should a group be allowed to match its own past (self-analog), or only other groups?
- How is movement scaled when matched series have different volatility?
- Single best match vs. similarity-weighted ensemble of top-k?
- Forecast horizon relative to window length.

**Rules for honest evaluation:**
- Walk-forward (rolling-origin) backtests only. Scaling, interpolation, and similarity must use only data available at each forecast origin (no lookahead).
- Baselines: last value (naive), drift, seasonal naive, and a simple ARIMA/ETS.
- Metrics: MAE/RMSE, directional accuracy, and calibration of any intervals.
- Report negative results too. Price series are close to random walks, so the method may show more value on other kinds of grouped series. Include at least one non-financial dataset in the demos.

## 8. Roadmap

The phases follow your seven updates (numbers in parentheses) but are reordered by dependency: the class and DataFrame foundation (Update 7) has to come first because every other feature plugs into it.

### Phase 0: Repository setup (complete, 2026-11-18)
- [x] Rotate exposed API keys; add `.env.example` and `.gitignore` (the first commit initially included `.venv` and `.ipynb_checkpoints`; these were untracked and removed from the repository)
- [x] `git init`, venv, `pyproject.toml` (hatchling, `src/` layout), editable install
- [x] pytest, ruff, and a basic CI workflow (GitHub Actions); ruff excludes `notebooks` and `*.md`
- [x] Download script and local S&P 100 Parquet (`examples/download_sp100.py`; data folder gitignored)
- [x] Move the cleaned prototype to `examples/stocks_alpaca.py` (reads the Parquet file)
- [ ] Fill in `README.md` and confirm the `LICENSE` choice (placeholders exist; needed for the build)

### Phase 1: Foundation: data model and class skeleton (Updates 1, 7)
- [ ] `data.py`: tidy-DataFrame schema, adapters (wide, MultiIndex), validation with clear errors
- [ ] `PiecewiseCorrelation` class with `fit`, `matrices_`, `combined_`, `weights_`
- [ ] Vectorized correlation; explicit weight lookup
- [ ] **Parity test:** reproduce the combined matrix of `examples/stocks_alpaca.py` through the class on a small committed fixture (synthetic, or a tiny sample if the data terms allow). Note that the example uses one shared time grid, while the original prototype built each symbol's series separately, so small differences from the original are expected; pin the chosen behavior in the test.
- [ ] Tests must not depend on `examples/data/`, which does not exist in CI
- **Done when:** the prototype's result matrix is reproduced through the class and the test passes.

### Phase 2: Missing data (Update 2)
- [ ] `missing.py` policies, defaults, `fill_report_`
- [ ] Test that structural gaps (overnight/weekends) are not interpolated
- [ ] Coverage threshold and warnings

### Phase 3: Built-in periods (Update 3)
- [ ] Period registry: seconds → yearly, plus weekday/hour-of-day/month-of-year labels
- [ ] Check that each requested period is not finer than the data cadence
- [ ] Pin pandas alias handling (≥2.2)

### Phase 4: Custom periods (Update 4)
- [ ] Resolve keys against user label columns, then the registry
- [ ] Documentation and examples for building label columns
- [ ] Tests for binned vs. label-based behavior

### Phase 5: Custom aggregation (Update 5)
- [ ] String and callable `agg`, applied to all periods
- [ ] Validate scalar output; tests with lambdas

### Phase 6: Clustering (Update 6a)
- [ ] `cluster()` with hierarchical default and a method/linkage option
- [ ] Dendrogram and heatmap plotting (optional extra)
- [ ] Synthetic planted-cluster benchmark and baseline comparisons

### Phase 7: Prediction (Update 6b)
- [ ] Expanding (no-lookahead) scaling
- [ ] Analog-forecast prototype in a notebook, then move to `predict.py`
- [ ] Walk-forward backtest harness and baselines

### Phase 8: Polish and portfolio presentation
- [ ] README with the idea, a quick-start, and result figures
- [ ] Docs site (MkDocs or Sphinx), docstrings in numpydoc style
- [ ] Type hints + mypy, 85%+ test coverage
- [ ] Two or three example notebooks (stocks, a non-financial dataset, a custom-period walkthrough)
- [ ] Publish to PyPI/TestPyPI (optional), tagged release v0.1.0

## 9. Testing and Validation Strategy

- **R parity:** export intermediate and final matrices from the original R code as CSV fixtures. Compare with `numpy.testing.assert_allclose`. Watch the known R/Python differences: 1-based vs 0-based indexing, `sd` (n−1) vs `np.std` (n), and different default interpolation/NaN handling.
- **Unit tests:** each module (validation, missing, periods, aggregation, scaling) in isolation.
- **Property tests:** symmetry, unit diagonal, bounds, invariance to group ordering, invariance to row shuffling.
- **Synthetic data generator** in `tests/` with known cluster structure for deterministic clustering tests.
- **Regression tests** for the bugs listed in §3.
- **No test depends on `examples/data/`.** That folder is gitignored and absent in CI. Tests use the synthetic generator or small committed fixtures.

## 10. Tooling and Conventions

- VS Code with Python, Pylance, and Ruff extensions; `.venv` selected as the interpreter.
- Notebooks for exploration use `%load_ext autoreload` / `%autoreload 2`; logic moves into `src/` once it works.
- Docstrings in numpydoc format; semantic versioning; conventional commit messages.
- Secrets only via environment variables (`.env`, never committed); nothing sensitive in notebooks, examples, or git history. `.env.example` is the committed template.
- Run the CI checks locally before pushing, with the venv activated: `ruff check .`, `ruff format --check .` (fix with `ruff format .`), and `pytest`.
- Ruff is configured in `pyproject.toml` to exclude `notebooks` and `*.md`, so exploratory code and documentation samples are not linted or reformatted.
- Keep one virtual environment per project. The `.gitignore` pattern `.venv*/` covers variants such as `.venv-1`.

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| Prediction shows little skill on price data | Define success up front; include non-financial datasets; report baselines honestly |
| Interpolation or scaling leaks future data | No-lookahead modes and explicit tests |
| Custom-period semantics confuse users | Clear distinction between binned and label-based periods in docs and error messages |
| Pandas alias changes between versions | Pin `pandas>=2.2`; centralize aliases in `periods.py` |
| Scope creep | Keep data fetching out of core; ship v0.1.0 after Phase 6 |
| Redistributing market data may violate the data provider's terms | Do not commit downloaded data (D8); provide the download script; use openly licensed datasets for committed fixtures and non-financial demos |

## 12. Decision Log

| ID | Decision | Status |
|----|----------|--------|
| D1 | Package name: `piecewise_correlation` (distribution name `piecewise-correlation`, repo: `piecewise-correlation-clustering`) | Adopted in `pyproject.toml` |
| D2 | Canonical internal format is a tidy long DataFrame; Series accepted only via conversion | Proposed |
| D3 | Default similarity is r² (matches prototype); `abs` and `signed` available | Proposed |
| D4 | How to combine matrices when a pair is NaN in some periods (re-normalize weights vs. flag) | **Open** |
| D5 | Default clustering method: hierarchical (average linkage) | Proposed |
| D6 | Can a group match its own past in prediction? | **Open** |
| D7 | Scaling default: `max` for clustering, expanding `max` for prediction | **Open** |
| D8 | Downloaded example data is not committed: `examples/data/` is gitignored and users generate it with `download_sp100.py` and their own free Alpaca keys | Decided |
| D9 | Ruff excludes `notebooks` and `*.md`; CI runs lint, format check, and tests on Python 3.10-3.12 | Decided |

## 13. Update Log

Add an entry each time scope changes, then update the relevant roadmap phase and decision rows.

| Date | Update | Affects | Notes |
|------|--------|---------|-------|
| 2026-11-17 | Initial plan created from the prototype and Updates 1-7 | All | Updates reordered by dependency (§8) |
| 2026-11-18 | Phase 0 completed. Examples switched to pre-downloaded Parquet data (2025-01-01 onward, S&P 100, hourly IEX); download script added; example data kept out of git; CI and ruff configured | §3, §4, §8 (Phase 0, Phase 1), §9, §10, §11, §12 (D1, D8, D9) | `.env` is only needed by `download_sp100.py`. Several §3 issues fixed in the example script; the class implementation is Phase 1 |

**Template for new updates:**

```
### Update N: <title>
- Goal / user-facing behavior:
- Affected modules:
- API changes (new/changed parameters):
- Tests to add:
- Roadmap phase: <new or existing>
- Open questions:
```
