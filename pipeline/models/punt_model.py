"""Expectation models for punts and returns, keyed on line of scrimmage and conditions.

Linear models on a cubic in field position plus wind and roof, fit on 2015-2025 punts and
returns. Each target (punting-team EPA, inside-20, touchback, fair catch, return yards,
returner EPA) gets its own coefficient set; kickoffs use only seasons under the current
kickoff rules.
"""
from __future__ import annotations

import numpy as np
import polars as pl

FEATURES = ["yl", "yl2", "yl3", "wind", "dome"]


def features(df: pl.DataFrame) -> pl.DataFrame:
    dome = pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64).fill_null(0)
    yl = pl.col("yardline_100").cast(pl.Float64).fill_null(60) / 100
    return df.with_columns(
        yl.alias("yl"), (yl ** 2).alias("yl2"), (yl ** 3).alias("yl3"),
        pl.when(dome == 1).then(0.0).otherwise(pl.col("wind").cast(pl.Float64).fill_null(8.0)).clip(0, 35).alias("wind") / 10,
        dome.alias("dome"),
    )


def fit(df: pl.DataFrame, target: str) -> dict:
    d = features(df.filter(pl.col(target).is_not_null()))
    X = np.column_stack([np.ones(d.height), d.select(FEATURES).to_numpy()])
    y = d[target].cast(pl.Float64).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(beta[0]), "coef": dict(zip(FEATURES, map(float, beta[1:]))),
            "n": int(d.height), "mean": float(y.mean())}


def predict(df: pl.DataFrame, model: dict) -> np.ndarray:
    d = features(df)
    return model["intercept"] + sum(d[f].fill_null(0).to_numpy() * c for f, c in model["coef"].items())
