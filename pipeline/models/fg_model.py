"""In-house make-probability models for field goals and extra points (trained on 2015-2025).

Logistic regression on distance, wind, temperature, surface, roof and altitude. Coefficients
are stored in baseline_stats.json so grading needs no sklearn model files.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

FEATURES = ["dist", "dist2", "dist3", "wind", "cold", "turf", "dome", "altitude"]


def features(df: pl.DataFrame) -> pl.DataFrame:
    dome = pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64).fill_null(0)
    return df.with_columns(
        (pl.col("kick_distance").cast(pl.Float64) / 50).alias("dist"),
        ((pl.col("kick_distance").cast(pl.Float64) / 50) ** 2).alias("dist2"),
        ((pl.col("kick_distance").cast(pl.Float64) / 50) ** 3).alias("dist3"),
        pl.when(dome == 1).then(0.0).otherwise(pl.col("wind").cast(pl.Float64).fill_null(8.0)).clip(0, 35).alias("wind") / 10,
        pl.when(dome == 1).then(0.0)
        .otherwise((50 - pl.col("temp").cast(pl.Float64).fill_null(60.0)).clip(0, 60)).alias("cold") / 10,
        (~pl.col("surface").fill_null("grass").str.contains("grass")).cast(pl.Float64).alias("turf"),
        dome.alias("dome"),
        (pl.col("home_team") == "DEN").cast(pl.Float64).fill_null(0).alias("altitude"),
    )


def fit(kicks: pl.DataFrame, made_col: str) -> dict:
    d = features(kicks.filter(pl.col("kick_distance").is_not_null()))
    X = d.select(FEATURES).to_numpy()
    y = d[made_col].to_numpy().astype(int)
    m = LogisticRegression(C=10.0, max_iter=2000).fit(X, y)
    return {"intercept": float(m.intercept_[0]), "coef": dict(zip(FEATURES, map(float, m.coef_[0]))),
            "n": int(len(y)), "base_rate": float(y.mean())}


def predict(df: pl.DataFrame, model: dict) -> pl.Series:
    d = features(df)
    z = model["intercept"] + sum(d[f].fill_null(0).to_numpy() * c for f, c in model["coef"].items())
    return pl.Series(1 / (1 + np.exp(-z)))
