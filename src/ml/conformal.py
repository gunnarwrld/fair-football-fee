"""Split-conformal prediction intervals and update-batch recentering (Phase 4).

Score: the absolute log residual |ln(1+y) - ln(1+y_hat)|, which equals |target - predicted ratio|.
An interval is therefore symmetric in log space and multiplicative in EUR:
(1 + value) in [(1 + y_hat) * exp(-q), (1 + y_hat) * exp(q)].

Mondrian variant: calibration rows are grouped by the model's own predicted ratio (quantile bins
fitted on the calibration set) and each bin gets its own quantile, so low- and high-value
predictions each get their nominal coverage instead of only the average.

Recentering: Transfermarkt revalues the league in update rounds ("batches"), and each round can
shift every player at once (Phase 3). A batch is a run of valuation dates with no gap longer than
`BATCH_GAP_DAYS`. A prediction may be shifted by the median residual of the most recent batch
that ended before the prediction's own batch began and has at least `MIN_BATCH_ROWS`
out-of-sample rows. Every row it reads is dated strictly before the prediction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BATCH_GAP_DAYS = 14
MIN_BATCH_ROWS = 50
LEVELS: tuple[float, ...] = (0.8, 0.9)
DISPLAY_LEVEL = 0.8
N_BINS = 4


def batch_table(dates: Iterable[Any]) -> pd.DataFrame:
    """One row per update batch: `batch`, `start`, `end`, `n` (rows dated inside it)."""
    d = pd.to_datetime(pd.Series(list(dates))).dropna()
    counts = d.value_counts().sort_index()
    gaps = counts.index.to_series().diff().dt.days.fillna(np.inf).to_numpy()
    ids = np.cumsum(gaps > BATCH_GAP_DAYS) - 1
    frame = pd.DataFrame({"date": counts.index, "n": counts.to_numpy(), "batch": ids})
    return (
        frame.groupby("batch")
        .agg(start=("date", "min"), end=("date", "max"), n=("n", "sum"))
        .reset_index()
    )


def batch_of(dates: Iterable[Any], table: pd.DataFrame) -> np.ndarray:
    """Batch id for each date, or -1 when the date lies outside every batch."""
    d = pd.to_datetime(pd.Series(list(dates))).to_numpy()
    starts, ends = table["start"].to_numpy(), table["end"].to_numpy()
    pos = np.searchsorted(starts, d, side="right") - 1
    inside = (pos >= 0) & (d <= ends[np.clip(pos, 0, None)])
    return np.where(inside, table["batch"].to_numpy()[np.clip(pos, 0, None)], -1)


def batch_offsets(
    query_dates: Iterable[Any],
    table: pd.DataFrame,
    ref_dates: Iterable[Any],
    ref_residuals: np.ndarray,
    min_rows: int = MIN_BATCH_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Recentering offset and source batch id (-1 = none, offset 0) for each query date.

    `ref_*` must be out-of-sample residuals (rows the model never trained on). A batch qualifies
    as a source when it has at least `min_rows` of them; its offset is their median.
    """
    q = pd.to_datetime(pd.Series(list(query_dates))).to_numpy()
    ref = pd.DataFrame({
        "batch": batch_of(ref_dates, table),
        "resid": np.asarray(ref_residuals, dtype=float),
    })
    stats = ref[ref["batch"] >= 0].groupby("batch")["resid"].agg(["median", "size"])
    source = table.merge(stats, left_on="batch", right_index=True)
    source = source[source["size"] >= min_rows].sort_values("end")

    own = batch_of(q, table)
    starts = table.set_index("batch")["start"]
    cutoff = np.where(own >= 0, starts.reindex(own).to_numpy(), q)  # the prediction's batch start
    pos = np.searchsorted(source["end"].to_numpy(), cutoff, side="left") - 1  # end strictly before
    has = pos >= 0
    offsets = np.where(has, source["median"].to_numpy()[np.clip(pos, 0, None)], 0.0) if len(source) else np.zeros(len(q))
    ids = np.where(has, source["batch"].to_numpy()[np.clip(pos, 0, None)], -1) if len(source) else np.full(len(q), -1)
    return offsets.astype(float), ids.astype(int)


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest score (finite-sample split-conformal quantile)."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if 1 <= k <= len(s) else float("inf")


@dataclass
class ConformalIntervals:
    """Per-bin conformal quantiles of |residual| on the predicted-ratio scale."""

    levels: list[float]
    edges: list[float]  # inner bin edges on the predicted ratio; empty = one global bin
    quantiles: dict[str, list[float]]  # str(level) -> quantile per bin
    n_calib: list[int]
    recentered: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, pred_ratio: np.ndarray, residual: np.ndarray, levels: Iterable[float] = LEVELS,
        n_bins: int = 1, recentered: bool = False,
    ) -> ConformalIntervals:
        pred = np.asarray(pred_ratio, dtype=float)
        scores = np.abs(np.asarray(residual, dtype=float))
        edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1)[1:-1]).tolist() if n_bins > 1 else []
        bins = np.searchsorted(edges, pred, side="right")
        levels = [float(lv) for lv in levels]
        quantiles = {str(lv): [conformal_quantile(scores[bins == b], lv) for b in range(n_bins)] for lv in levels}
        n_calib = [int((bins == b).sum()) for b in range(n_bins)]
        return cls(levels, edges, quantiles, n_calib, recentered)

    def bins(self, pred_ratio: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(pred_ratio, dtype=float), side="right")

    def halfwidth(self, pred_ratio: np.ndarray, level: float) -> np.ndarray:
        return np.asarray(self.quantiles[str(float(level))])[self.bins(pred_ratio)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalIntervals:
        return cls(**data)


def coverage_by(covered: np.ndarray, groups: Iterable[Any]) -> list[dict[str, Any]]:
    """Coverage and row count per group, in group order."""
    df = pd.DataFrame({"g": list(groups), "c": np.asarray(covered, dtype=float)})
    out = df.groupby("g", sort=True)["c"].agg(["size", "mean"]).reset_index()
    return [{"group": str(r.g), "n": int(r.size), "coverage": float(r.mean)} for r in out.itertuples()]
