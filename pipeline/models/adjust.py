"""Step 2 — context adjustment.

Ridge regression of a play outcome on the player, the opposing unit and the situation
(down, distance, field position, score, weather, roof, home). The fitted opponent and
situation effects are removed from each play, leaving an opponent- and situation-adjusted
value whose per-player mean is that player's adjusted rate. (Averaging residualized plays and
then shrinking in step 3 is equivalent to reading the player coefficient, but keeps the
per-game structure needed for game grades and the bootstrap.)
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import Ridge

SITUATION = [
    "down1", "down2", "down3", "down4", "ydstogo_c", "yl", "yl2", "redzone", "goal",
    "score_c", "late", "wind_c", "dome", "is_home",
]


def situation_frame(df: pl.DataFrame, extra: list[str] | None = None) -> pl.DataFrame:
    out = df.with_columns(
        *[(pl.col("down") == d).cast(pl.Float64).fill_null(0).alias(f"down{d}") for d in (1, 2, 3, 4)],
        pl.col("ydstogo").fill_null(10).clip(1, 20).cast(pl.Float64).alias("ydstogo_c"),
        (pl.col("yardline_100").fill_null(50) / 100).alias("yl"),
        ((pl.col("yardline_100").fill_null(50) / 100) ** 2).alias("yl2"),
        (pl.col("yardline_100") <= 20).cast(pl.Float64).fill_null(0).alias("redzone"),
        (pl.col("yardline_100") <= pl.col("ydstogo")).cast(pl.Float64).fill_null(0).alias("goal"),
        (pl.col("score_differential").fill_null(0).clip(-21, 21) / 7).alias("score_c"),
        ((pl.col("qtr") >= 4) & (pl.col("score_differential").abs() <= 8)).cast(pl.Float64).fill_null(0).alias("late"),
        pl.when(pl.col("is_dome") == 1).then(0.0).otherwise(pl.col("wind").cast(pl.Float64).fill_null(8.0)).clip(0, 30).alias("wind_c"),
        pl.col("is_dome").fill_null(0).alias("dome"),
        pl.col("is_home").fill_null(0.5),
    )
    cols = SITUATION + (extra or [])
    return out.with_columns([pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).mean()).fill_null(0) for c in cols])


def adjust(df: pl.DataFrame, value: str, player: str, opp: str, extra: list[str] | None = None,
           alpha: float = 25.0, out: str | None = None) -> pl.DataFrame:
    """Return df with `<out>` = value minus fitted opponent and situation effects.

    The league-average level is preserved: only deviations from the average opponent and the
    average situation are removed, so adjusted EPA stays on the EPA scale.
    """
    out = out or f"{value}_adj"
    d = situation_frame(df.filter(pl.col(value).is_not_null()), extra)
    if d.height < 50:
        return d.with_columns(pl.col(value).alias(out))
    cols = SITUATION + (extra or [])
    X_sit = d.select(cols).to_numpy().astype(float)
    mu, sd = X_sit.mean(0), X_sit.std(0)
    sd[sd == 0] = 1
    X_sit = (X_sit - mu) / sd

    def onehot(col: str):
        codes = d[col].fill_null("_none").cast(pl.Categorical).to_physical().to_numpy()
        n = codes.max() + 1
        return sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), n)), codes

    P, _ = onehot(player)
    O, ocodes = onehot(opp)
    X = sp.hstack([P, O, sp.csr_matrix(X_sit)]).tocsr()
    y = d[value].cast(pl.Float64).to_numpy()
    w = d["w"].to_numpy() if "w" in d.columns else np.ones(len(y))
    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
    np_, no_ = P.shape[1], O.shape[1]
    opp_coef = m.coef_[np_:np_ + no_]
    sit_coef = m.coef_[np_ + no_:]
    opp_eff = opp_coef[ocodes] - np.average(opp_coef[ocodes], weights=w)
    sit_eff = X_sit @ sit_coef
    sit_eff = sit_eff - np.average(sit_eff, weights=w)
    return d.with_columns(pl.Series(out, y - opp_eff - sit_eff))
