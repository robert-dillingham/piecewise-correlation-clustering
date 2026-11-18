"""Piecewise correlation on the S&P 100, step by step (prototype logic).

Reads the Parquet written by download_sp100.py. This reproduces the original
prototype's calculation; the PiecewiseCorrelation class (Phase 1) will replace it.

Known issues kept on purpose so the Phase 1 parity test has a baseline:
  - scaling divides by the full-sample max (lookahead; fine for clustering only)
  - empty time bins are interpolated (to be fixed in Phase 2)
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "sp100_hourly.parquet"

# period name -> (weight, pandas frequency alias)
PERIODS = {
    "hourly": (4, "h"),
    "daily": (3, "D"),
    "weekly": (2, "W"),
    "monthly": (1, "ME"),
}


def load_wide(path: Path) -> pd.DataFrame:
    """Tidy parquet -> wide frame: index = New York time, columns = symbols, values = high."""
    df = pd.read_parquet(path)
    wide = df.pivot(index="timestamp", columns="symbol", values="high")
    if wide.index.tz is None:
        wide.index = wide.index.tz_localize("UTC")
    return wide.tz_convert("America/New_York")


def main() -> None:
    wide = load_wide(DATA)

    # standardize: x / max(x) per symbol (same as 1 + (x - max) / max)
    standardized = wide / wide.max()

    # normalize weights, looked up by period name
    total = sum(weight for weight, _ in PERIODS.values())
    weights = {name: weight / total for name, (weight, _) in PERIODS.items()}

    # one r^2 matrix per period
    matrices = {}
    for name, (_, freq) in PERIODS.items():
        summary = standardized.resample(freq).mean().interpolate()
        matrices[name] = summary.corr() ** 2

    # weighted average across periods
    combined = sum(weights[name] * matrices[name] for name in PERIODS)

    print(f"{combined.shape[0]} symbols, {len(wide):,} timestamps")
    # upper triangle only, so each pair appears once and the diagonal is skipped
    rows, cols = np.triu_indices(len(combined), k=1)
    pairs = pd.Series(
        combined.to_numpy()[rows, cols],
        index=pd.MultiIndex.from_arrays([combined.index[rows], combined.columns[cols]]),
    )
    top = pairs.sort_values(ascending=False).head(10)
    print("\nMost similar pairs:")
    print(top.round(3).to_string())


if __name__ == "__main__":
    main()
