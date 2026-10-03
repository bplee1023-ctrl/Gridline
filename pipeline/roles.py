"""Assign each player one primary position group per season.

Primary source: depth charts, weighted by how often the player is listed (the scheme matters:
in a 3-4 front ESPN's 'WLB/SLB' are edge rushers and 'LDE/RDE' are interior linemen).
Fallback: PFR snap-count position, then the nflverse players table.
Hybrid fix-up: a 'LB' who generates pressure like an edge rusher is moved to EDGE.
Final word: config/position_overrides.yaml.
"""
from __future__ import annotations

import polars as pl

from .config import overrides_cfg
from .ledger import Ctx, player_game_frame

OFF_SIMPLE = {
    "QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "LWR": "WR", "RWR": "WR",
    "SWR": "WR", "TE": "TE", "LTE": "TE", "RTE": "TE", "F": "TE", "LT": "OT", "RT": "OT",
    "T": "OT", "OT": "OT", "LOT": "OT", "ROT": "OT", "LG": "IOL", "RG": "IOL", "G": "IOL",
    "C": "IOL", "OG": "IOL", "OL": "IOL", "PK": "K", "K": "K", "P": "P",
}
DEF_OLD = {
    "EDGE": "EDGE", "DE": "EDGE", "LDE": "EDGE", "RDE": "EDGE", "LE": "EDGE", "RE": "EDGE",
    "RUSH": "EDGE", "LOLB": "EDGE", "ROLB": "EDGE", "OLB": "EDGE", "WRE": "EDGE", "JACK": "EDGE",
    "DT": "IDL", "LDT": "IDL", "RDT": "IDL", "NT": "IDL", "N": "IDL", "DL": "IDL", "UT": "IDL",
    "ILB": "LB", "LILB": "LB", "RILB": "LB", "MLB": "LB", "MIKE": "LB", "LB": "LB", "WLB": "LB",
    "SLB": "LB", "SAM": "LB", "WIL": "LB", "WILL": "LB",
    "CB": "CB", "LCB": "CB", "RCB": "CB", "NB": "CB", "NCB": "CB", "NICK": "CB", "NKL": "CB",
    "NDB": "CB", "DB": "CB", "NICKE": "CB",
    "S": "S", "FS": "S", "SS": "S", "SAF": "S",
}
# ESPN depth charts (2025+): meaning depends on the base front
DEF_34 = {"LDE": "IDL", "RDE": "IDL", "NT": "IDL", "LDT": "IDL", "RDT": "IDL", "WLB": "EDGE",
          "SLB": "EDGE", "LOLB": "EDGE", "ROLB": "EDGE", "LILB": "LB", "RILB": "LB", "MLB": "LB"}
DEF_43 = {"LDE": "EDGE", "RDE": "EDGE", "LDT": "IDL", "RDT": "IDL", "NT": "IDL", "WLB": "LB",
          "MLB": "LB", "SLB": "LB", "LILB": "LB", "RILB": "LB"}
PFR_POS = {
    "QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OT", "OT": "OT",
    "G": "IOL", "C": "IOL", "OL": "IOL", "DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL",
    "DL": "IDL", "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S",
    "FS": "S", "SS": "S", "SAF": "S", "K": "K", "P": "P",
}
SKIP = {"KR", "PR", "H", "LS", "KO", "KOR", "PR/KR"}


def _depth_counts(depth: pl.DataFrame) -> pl.DataFrame:
    if depth.height == 0:
        return pl.DataFrame(schema={"gsis_id": pl.String, "position": pl.String, "n": pl.Float64})
    if "pos_abb" in depth.columns:  # ESPN format
        d = depth.filter(pl.col("gsis_id").is_not_null() & ~pl.col("pos_abb").is_in(list(SKIP)))
        grp = pl.col("pos_grp").fill_null("")
        abb = pl.col("pos_abb")
        mapped = (
            pl.when(grp.str.contains("3-4")).then(abb.replace_strict(DEF_34, default=None))
            .when(grp.str.contains("4-3")).then(abb.replace_strict(DEF_43, default=None))
            .otherwise(None)
        )
        mapped = pl.coalesce(mapped, abb.replace_strict(OFF_SIMPLE, default=None),
                             abb.replace_strict(DEF_OLD, default=None))
        # starters count more than backups
        wt = 1.0 / pl.col("pos_rank").cast(pl.Float64).fill_null(1.0).clip(1, 4)
    else:  # weekly format (2001-2024)
        d = depth.filter(pl.col("gsis_id").is_not_null())
        pos = pl.col("depth_position").str.strip_chars().str.to_uppercase()
        d = d.filter(~pos.is_in(list(SKIP)))
        mapped = pl.coalesce(pos.replace_strict(OFF_SIMPLE, default=None),
                             pos.replace_strict(DEF_OLD, default=None))
        wt = 1.0 / pl.col("depth_team").cast(pl.Float64, strict=False).fill_null(1.0).clip(1, 4)
    return (d.with_columns(mapped.alias("position"), wt.alias("wt"))
            .drop_nulls("position")
            .group_by("gsis_id", "position").agg(pl.col("wt").sum().alias("n")))


def assign(ctx: Ctx) -> pl.DataFrame:
    pg = player_game_frame(ctx)
    tot = pg.group_by("gsis_id").agg(
        pl.col("off_snaps").sum(), pl.col("def_snaps").sum(), pl.col("st_snaps").sum(),
        pl.col("def_pass_snaps").sum(),
        pl.col("pfr_pos").sort_by(["off_snaps", "game_id"]).last().alias("pfr_pos_off"),
        pl.col("pfr_pos").sort_by(["def_snaps", "game_id"]).last().alias("pfr_pos_def"),
    )
    dc = _depth_counts(ctx.depth)
    best = dc.sort(["gsis_id", "n", "position"], descending=[False, True, False]).unique("gsis_id", keep="first", maintain_order=True).select("gsis_id", pl.col("position").alias("depth_pos"))

    players = (
        pl.concat([tot.select("gsis_id"), best.select("gsis_id")]).unique()
        .join(tot, on="gsis_id", how="left").join(best, on="gsis_id", how="left")
        .join(ctx.players.select("gsis_id", "nfl_position"), on="gsis_id", how="left")
        .with_columns(pl.col(c).fill_null(0) for c in ("off_snaps", "def_snaps", "st_snaps", "def_pass_snaps"))
    )
    off_groups = ["QB", "RB", "WR", "TE", "OT", "IOL"]
    def_groups = ["EDGE", "IDL", "LB", "CB", "S"]
    pfr_fallback = pl.when(pl.col("off_snaps") >= pl.col("def_snaps")) \
        .then(pl.col("pfr_pos_off")).otherwise(pl.col("pfr_pos_def")).replace_strict(PFR_POS, default=None)
    nfl_fallback = pl.col("nfl_position").replace_strict(PFR_POS, default=None)
    pos = pl.coalesce("depth_pos", pfr_fallback, nfl_fallback)
    # side sanity: depth chart says offense but the player plays defense (or vice versa)
    side_off = pl.col("off_snaps") > 2 * pl.col("def_snaps") + 20
    side_def = pl.col("def_snaps") > 2 * pl.col("off_snaps") + 20
    pos = (pl.when(side_def & pos.is_in(off_groups)).then(pl.coalesce(pfr_fallback, pos))
           .when(side_off & pos.is_in(def_groups)).then(pl.coalesce(pfr_fallback, pos))
           .otherwise(pos))
    players = players.with_columns(pos.alias("position")).drop_nulls("position")

    # LB -> EDGE when they pressure like an edge rusher
    if ctx.pfr_def.height:
        press = ctx.pfr_def.group_by("gsis_id").agg(pl.col("def_pressures").sum().alias("press"))
        players = players.join(press, on="gsis_id", how="left").with_columns(
            pl.when((pl.col("position") == "LB") & (pl.col("def_pass_snaps") >= 80)
                    & (pl.col("press").fill_null(0) / pl.col("def_pass_snaps") >= 0.04))
            .then(pl.lit("EDGE")).otherwise(pl.col("position")).alias("position"))

    ov = overrides_cfg()
    if ov:
        players = players.with_columns(
            pl.col("gsis_id").replace_strict(ov, default=pl.col("position")).alias("position"))
    return players.select("gsis_id", "position").sort("gsis_id")
