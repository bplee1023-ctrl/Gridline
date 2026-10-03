from __future__ import annotations

import polars as pl

LONG_SCHEMA = {"gsis_id": pl.String, "game_id": pl.String, "metric": pl.String,
               "num": pl.Float64, "den": pl.Float64}


def emit(df: pl.DataFrame, id_col: str, name: str, num: pl.Expr, den: pl.Expr) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(schema=LONG_SCHEMA)
    # evaluate row-wise first so literal numerators/denominators broadcast to one per row
    return (df.filter(pl.col(id_col).is_not_null())
            .with_columns(num.cast(pl.Float64).fill_null(0).alias("_num"),
                          den.cast(pl.Float64).fill_null(0).alias("_den"))
            .group_by(pl.col(id_col).alias("gsis_id"), "game_id")
            .agg(pl.col("_num").sum().alias("num"), pl.col("_den").sum().alias("den"))
            .filter(pl.col("den") > 0)
            .with_columns(pl.lit(name).alias("metric"))
            .select(list(LONG_SCHEMA)))


def concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [f.select(list(LONG_SCHEMA)).cast(LONG_SCHEMA) for f in frames if f is not None and f.height]
    return pl.concat(frames) if frames else pl.DataFrame(schema=LONG_SCHEMA)


def scrimmage(plays: pl.DataFrame) -> pl.DataFrame:
    """Real offensive snaps: passes and runs, no kneels/spikes/two-point tries."""
    return plays.filter(
        pl.col("play_type").is_in(["pass", "run"]) & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1) & (pl.col("two_point_attempt").fill_null(0) != 1)
        & pl.col("epa").is_not_null()
    )


def tacklers(plays: pl.DataFrame) -> pl.DataFrame:
    """Long table of (play, defender, credit) for tackles; solo = 1, assisted = 1/2."""
    solo = ["solo_tackle_1_player_id", "solo_tackle_2_player_id", "tackle_with_assist_1_player_id",
            "tackle_with_assist_2_player_id"]
    assist = ["assist_tackle_1_player_id", "assist_tackle_2_player_id"]
    frames = []
    for cols, credit in ((solo, 1.0), (assist, 0.5)):
        for c in cols:
            frames.append(plays.select("game_id", "play_id", pl.col(c).alias("tackler"))
                          .drop_nulls("tackler").with_columns(pl.lit(credit).alias("credit")))
    t = pl.concat(frames).group_by("game_id", "play_id", "tackler").agg(pl.col("credit").max())
    tfl = pl.concat([plays.select("game_id", "play_id", pl.col(c).alias("tackler")).drop_nulls("tackler")
                     for c in ("tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id")]).unique() \
        .with_columns(pl.lit(1.0).alias("tfl"))
    return t.join(tfl, on=["game_id", "play_id", "tackler"], how="full", coalesce=True).with_columns(
        pl.col("credit").fill_null(1.0), pl.col("tfl").fill_null(0.0))


def penalties(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter((pl.col("penalty") == 1) & pl.col("penalty_player_id").is_not_null()) \
        .select("game_id", "play_id", pl.col("penalty_player_id").alias("gsis_id"),
                pl.col("penalty_yards").fill_null(0).cast(pl.Float64).alias("pen_yds"), "penalty_type")
