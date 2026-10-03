"""Step 3 — stabilization constants.

For each metric, k is the sample at which the metric is half signal. Estimated from split-half
(odd vs. even games) reliability across player-seasons:
    r_half = corr(rate_odd, rate_even);  r_full = 2 r_half / (1 + r_half)   (Spearman-Brown)
    reliability(n) = n / (n + k)  =>  k = n_bar * (1 - r_full) / r_full
"""
from __future__ import annotations

import numpy as np
import polars as pl

K_MIN, K_MAX = 1.0, 5000.0


def shrink(rate, n, k, mu):
    return (n * rate + k * mu) / (n + k)


def estimate_k(pg: pl.DataFrame, min_den_half: float) -> float | None:
    """pg: rows (season, gsis_id, game_idx, num, den) for one metric at one position."""
    h = (pg.with_columns((pl.col("game_idx") % 2).alias("half"))
         .group_by("season", "gsis_id", "half").agg(pl.col("num").sum(), pl.col("den").sum())
         .filter(pl.col("den") >= min_den_half)
         .with_columns((pl.col("num") / pl.col("den")).alias("rate")))
    wide = h.pivot(on="half", index=["season", "gsis_id"], values=["rate", "den"])
    need = {"rate_0", "rate_1", "den_0", "den_1"}
    if not need.issubset(wide.columns):
        return None
    wide = wide.drop_nulls(list(need))
    if wide.height < 15:
        return None
    a, b = wide["rate_0"].to_numpy(), wide["rate_1"].to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    nbar = float((wide["den_0"] + wide["den_1"]).mean())
    if not np.isfinite(r) or r <= 0.02:
        return K_MAX if nbar * 50 > K_MAX else nbar * 50
    r_full = 2 * r / (1 + r)
    return float(np.clip(nbar * (1 - r_full) / r_full, K_MIN, K_MAX))
