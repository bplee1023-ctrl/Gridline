"""Estimated blocking: OL pass protection and run blocking, WR/TE run blocking, TE/RB pass pro.

No free source credits individual blockers in-season, so these are apportioned from team
results (design doc, "The gap that shapes the design"):
  * Pass protection — team pressures and sacks over expected, controlling for time to throw,
    number of rushers, blitz, play action and the opponent's pass rush. Pressures/sacks by
    EDGE defenders are charged to tackles, by interior defenders to guards and center, by
    linebackers/DBs (blitzers) to the backs and tight ends; split by snap share.
  * Run blocking — opponent/situation/box-adjusted run EPA, adjusted line yards and success,
    net of the ball carrier, credited to the depth-chart lineman at the run gap.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..ledger import Ctx, player_game_frame
from ._util import concat, emit, scrimmage
from .qb import rushes

SLOTS = ["LT", "LG", "C", "RG", "RT"]
GAP_CREDIT = {
    ("left", "end"): {"LT": 1.0},
    ("left", "tackle"): {"LT": 0.5, "LG": 0.5},
    ("left", "guard"): {"LG": 0.5, "C": 0.5},
    ("middle", None): {"C": 0.5, "LG": 0.25, "RG": 0.25},
    ("right", "guard"): {"RG": 0.5, "C": 0.5},
    ("right", "tackle"): {"RT": 0.5, "RG": 0.5},
    ("right", "end"): {"RT": 1.0},
}


def _team_pressure_table(ctx: Ctx) -> pl.DataFrame:
    """Offense team-game: dropbacks, pressures and sacks by rusher group, plus controls."""
    pos = ctx.positions
    grp = (pl.when(pl.col("position") == "EDGE").then(pl.lit("edge"))
           .when(pl.col("position") == "IDL").then(pl.lit("int")).otherwise(pl.lit("off")))
    tg = ctx.team_games.select("game_id", "team", "opp", "dropbacks").filter(pl.col("dropbacks") > 0)

    if ctx.pfr_def.height:
        pr = ctx.pfr_def.join(pos, on="gsis_id", how="left").with_columns(grp.alias("g")) \
            .group_by("game_id", pl.col("opponent").alias("team")).agg(
                *[pl.col("def_pressures").filter(pl.col("g") == g).sum().alias(f"{g}_press") for g in ("edge", "int", "off")])
        tg = tg.join(pr, on=["game_id", "team"], how="left")
    else:
        tg = tg.with_columns(*[pl.lit(None, pl.Float64).alias(f"{g}_press") for g in ("edge", "int", "off")])

    db = scrimmage(ctx.plays).filter(pl.col("qb_dropback") == 1)
    sk = pl.concat([
        db.select("game_id", pl.col("posteam").alias("team"), pl.col("sack_player_id").alias("gsis_id"), pl.lit(1.0).alias("v")),
        db.select("game_id", pl.col("posteam").alias("team"), pl.col("half_sack_1_player_id").alias("gsis_id"), pl.lit(0.5).alias("v")),
        db.select("game_id", pl.col("posteam").alias("team"), pl.col("half_sack_2_player_id").alias("gsis_id"), pl.lit(0.5).alias("v")),
    ]).drop_nulls("gsis_id").join(pos, on="gsis_id", how="left").with_columns(grp.alias("g")) \
        .group_by("game_id", "team").agg(
            *[pl.col("v").filter(pl.col("g") == g).sum().alias(f"{g}_sacks") for g in ("edge", "int", "off")])
    tg = tg.join(sk, on=["game_id", "team"], how="left")

    ftn = db.filter(pl.col("charted")).group_by("game_id", pl.col("posteam").alias("team")).agg(
        pl.col("is_play_action").mean().alias("pa_rate"),
        pl.col("n_pass_rushers").cast(pl.Float64).mean().alias("rushers"),
        (pl.col("n_blitzers").fill_null(0) > 0).mean().alias("blitz_rate"))
    tg = tg.join(ftn, on=["game_id", "team"], how="left")
    if ctx.ngs_pass.height:
        ttt = ctx.ngs_pass.group_by("game_id", pl.col("team_abbr").alias("team")).agg(
            ((pl.col("avg_time_to_throw") * pl.col("attempts")).sum() / pl.col("attempts").sum()).alias("ttt"))
        tg = tg.join(ttt, on=["game_id", "team"], how="left")
    else:
        tg = tg.with_columns(pl.lit(None, pl.Float64).alias("ttt"))
    tg = tg.with_columns(pl.col(c).fill_null(0) for c in
                         ("edge_press", "int_press", "off_press", "edge_sacks", "int_sacks", "off_sacks"))
    # opponent pass-rush strength: what this defense generates per dropback over the season
    for g in ("edge", "int", "off"):
        opp_str = tg.group_by(pl.col("opp")).agg(
            ((pl.col(f"{g}_press").sum() + pl.col(f"{g}_sacks").sum()) / pl.col("dropbacks").sum()).alias(f"opp_{g}"))
        tg = tg.join(opp_str, on="opp", how="left")
    return tg


def _over_expected(tg: pl.DataFrame, target: str, opp_col: str) -> pl.Series:
    feats = ["ttt", "pa_rate", "rushers", "blitz_rate", opp_col]
    X = tg.select([pl.col(f).cast(pl.Float64).fill_null(pl.col(f).cast(pl.Float64).mean()).fill_null(0) for f in feats]).to_numpy()
    y = (tg[target] / tg["dropbacks"]).to_numpy()
    wt = tg["dropbacks"].to_numpy().astype(float)
    if len(y) < 30:
        exp = np.full(len(y), np.average(y, weights=wt))
    else:
        Xc = np.column_stack([np.ones(len(y)), (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)])
        W = np.sqrt(wt)[:, None]
        beta, *_ = np.linalg.lstsq(Xc * W, y * W[:, 0], rcond=None)
        exp = Xc @ beta
    return pl.Series(tg[target].to_numpy() - exp * wt)


def _ol_slots(ctx: Ctx, pg: pl.DataFrame) -> pl.DataFrame:
    """(game_id, team, slot, gsis_id) for the five starting linemen of every team-game."""
    games = ctx.games.select("game_id", "week", "gameday")
    tgames = pl.concat([games.join(ctx.games.select("game_id", pl.col(t).alias("team")), on="game_id")
                        for t in ("home_team", "away_team")])
    d = ctx.depth
    slots = pl.DataFrame(schema={"game_id": pl.String, "team": pl.String, "slot": pl.String, "gsis_id": pl.String})
    if d.height and "pos_abb" in d.columns:
        dd = d.filter(pl.col("pos_abb").is_in(SLOTS) & (pl.col("pos_rank") == 1) & pl.col("gsis_id").is_not_null()) \
            .select("team", pl.col("pos_abb").alias("slot"), "gsis_id",
                    pl.col("dt").str.slice(0, 10).str.to_date().alias("d")).unique(keep="last", maintain_order=True, subset=["team", "slot", "d"]).sort("d")
        tg = tgames.with_columns(pl.col("gameday").str.to_date().alias("d")).sort("d")
        frames = []
        for s in SLOTS:
            frames.append(tg.join_asof(dd.filter(pl.col("slot") == s), on="d", by="team", strategy="backward")
                          .select("game_id", "team", pl.lit(s).alias("slot"), "gsis_id"))
        slots = pl.concat(frames).drop_nulls("gsis_id")
    elif d.height and "depth_position" in d.columns:
        dd = d.filter(pl.col("depth_position").str.strip_chars().is_in(SLOTS)
                      & (pl.col("depth_team").cast(pl.String) == "1") & pl.col("gsis_id").is_not_null()) \
            .select(pl.col("club_code").alias("team"), pl.col("week").cast(pl.Int32),
                    pl.col("depth_position").str.strip_chars().alias("slot"), "gsis_id").unique(keep="first", maintain_order=True, subset=["team", "week", "slot"])
        slots = tgames.join(dd, on=["team", "week"], how="inner").select("game_id", "team", "slot", "gsis_id")

    # keep only starters who actually played most of the game; fill vacancies from the OL who did
    ol = pg.join(ctx.positions.filter(pl.col("position").is_in(["OT", "IOL"])), on="gsis_id") \
        .filter(pl.col("off_snaps") >= 0.5 * pl.col("team_off_snaps"))
    valid = slots.join(ol.select("game_id", "team", "gsis_id"), on=["game_id", "team", "gsis_id"], how="inner") \
        .sort("game_id", "team", "slot").unique(keep="first", maintain_order=True, subset=["game_id", "team", "gsis_id"])
    rows = valid.to_dicts()
    taken = {(r["game_id"], r["team"], r["slot"]) for r in rows}
    used = {(r["game_id"], r["team"], r["gsis_id"]) for r in rows}
    for (gid, team), grp in ol.sort(["game_id", "team", "off_snaps", "gsis_id"], descending=[False, False, True, False]) \
            .group_by(["game_id", "team"], maintain_order=True):
        if not any((gid, team, s) in taken for s in SLOTS):
            continue  # no depth chart for this team-game: handled by equal-credit fallback
        for r in grp.iter_rows(named=True):
            if (gid, team, r["gsis_id"]) in used:
                continue
            pref = ["LT", "RT"] if r["position"] == "OT" else ["C", "LG", "RG"]
            for s in pref + SLOTS:
                if (gid, team, s) not in taken:
                    rows.append({"game_id": gid, "team": team, "slot": s, "gsis_id": r["gsis_id"]})
                    taken.add((gid, team, s)); used.add((gid, team, r["gsis_id"]))
                    break
    return pl.DataFrame(rows, schema={"game_id": pl.String, "team": pl.String, "slot": pl.String, "gsis_id": pl.String}), ol


def metrics(ctx: Ctx) -> pl.DataFrame:
    pg = player_game_frame(ctx)
    pos = ctx.positions
    out = []

    # ---------------- pass protection
    tg = _team_pressure_table(ctx)
    if tg.height:
        tg = tg.with_columns(
            _over_expected(tg, "edge_press", "opp_edge").alias("edge_press_oe"),
            _over_expected(tg, "int_press", "opp_int").alias("int_press_oe"),
            _over_expected(tg, "off_press", "opp_off").alias("off_press_oe"),
            _over_expected(tg, "edge_sacks", "opp_edge").alias("edge_sacks_oe"),
            _over_expected(tg, "int_sacks", "opp_int").alias("int_sacks_oe"),
        )
        pp = pg.join(pos, on="gsis_id").filter(pl.col("off_snaps") > 0) \
            .join(tg.select("game_id", "team", "edge_press_oe", "int_press_oe", "off_press_oe", "edge_sacks_oe",
                            "int_sacks_oe", "dropbacks"), on=["game_id", "team"], how="inner")
        for group, key in ((["OT"], "edge"), (["IOL"], "int")):
            g = pp.filter(pl.col("position").is_in(group))
            g = g.with_columns((pl.col("off_snaps") / pl.col("off_snaps").sum().over("game_id", "team")).alias("share"))
            out.append(emit(g, "gsis_id", "pp_press_oe", -pl.col(f"{key}_press_oe") * pl.col("share"), pl.col("pass_snaps") / g_size(group)))
            out.append(emit(g, "gsis_id", "pp_sack_oe", -pl.col(f"{key}_sacks_oe") * pl.col("share"), pl.col("pass_snaps") / g_size(group)))
        g = pp.filter(pl.col("position").is_in(["RB", "TE"]))
        g = g.with_columns((pl.col("off_snaps") / pl.col("off_snaps").sum().over("game_id", "team")).alias("share"))
        out.append(emit(g, "gsis_id", "offball_press_oe", -pl.col("off_press_oe") * pl.col("share"), pl.col("pass_snaps")))

    # RB: sacks over expected on blitz dropbacks while on the field
    db = scrimmage(ctx.plays).filter((pl.col("qb_dropback") == 1) & pl.col("charted") & (pl.col("n_blitzers").fill_null(0) > 0))
    if db.height:
        db = db.with_columns(pl.col("n_pass_rushers").fill_null(5).clip(4, 7).alias("nr"))
        db = db.join(db.group_by("nr").agg(pl.col("sack").mean().alias("exp")), on="nr")
        team_b = db.group_by("game_id", pl.col("posteam").alias("team")).agg(
            (pl.col("sack") - pl.col("exp")).sum().alias("bsoe"), pl.len().alias("blitz_db"))
        rb = pg.join(pos.filter(pl.col("position") == "RB"), on="gsis_id").filter(pl.col("off_snaps") > 0) \
            .join(team_b, on=["game_id", "team"], how="inner").join(
                ctx.team_games.select("game_id", "team", pl.col("dropbacks").alias("tdb")), on=["game_id", "team"])
        rb = rb.with_columns((pl.col("off_snaps") / pl.col("off_snaps").sum().over("game_id", "team")).alias("share"))
        out.append(emit(rb, "gsis_id", "rb_blitz_sack_oe", -pl.col("bsoe") * pl.col("share"),
                        pl.col("pass_snaps") * pl.col("blitz_db") / pl.col("tdb")))

    # ---------------- run blocking
    r = rushes(ctx).filter(pl.col("qb_scramble") != 1)
    # remove the ball carrier's own (shrunk) effect so linemen are not credited for the back
    k = 60.0
    lg = {v: float(r[f"{v}_adj"].mean() or 0) for v in ("epa", "success", "aly")}
    carrier = r.group_by("rusher_player_id").agg(
        *[(((pl.col(f"{v}_adj") - lg[v]).sum()) / (pl.len() + k)).alias(f"c_{v}") for v in ("epa", "success", "aly")])
    r = r.join(carrier, on="rusher_player_id", how="left").with_columns(
        *[(pl.col(f"{v}_adj") - pl.col(f"c_{v}").fill_null(0)).alias(f"ol_{v}") for v in ("epa", "success", "aly")])
    slots, ol_played = _ol_slots(ctx, pg)
    credit_rows = []
    for (loc, gap), cr in GAP_CREDIT.items():
        for s, c in cr.items():
            credit_rows.append({"run_location": loc, "run_gap": gap, "slot": s, "credit": c})
    credit = pl.DataFrame(credit_rows, schema={"run_location": pl.String, "run_gap": pl.String, "slot": pl.String, "credit": pl.Float64})
    rr = r.with_columns(pl.when(pl.col("run_location") == "middle").then(None).otherwise(pl.col("run_gap")).alias("run_gap"),
                        pl.col("posteam").alias("team"))
    by_slot = rr.join(credit, on=["run_location", "run_gap"], how="inner", nulls_equal=True) \
        .join(slots, on=["game_id", "team", "slot"], how="inner")
    # team-games without a depth chart: equal credit across linemen who played half the snaps
    have = slots.select("game_id", "team").unique()
    eq = rr.join(have, on=["game_id", "team"], how="anti").join(
        ol_played.select("game_id", "team", "gsis_id"), on=["game_id", "team"], how="inner") \
        .with_columns((1.0 / pl.len().over("game_id", "play_id")).alias("credit"))
    runs_c = pl.concat([by_slot.select("gsis_id", "game_id", "w", "credit", "ol_epa", "ol_success", "ol_aly"),
                        eq.select("gsis_id", "game_id", "w", "credit", "ol_epa", "ol_success", "ol_aly")])
    wc = pl.col("w") * pl.col("credit")
    out.append(emit(runs_c, "gsis_id", "ol_run_epa", wc * pl.col("ol_epa"), wc))
    out.append(emit(runs_c, "gsis_id", "ol_aly", wc * pl.col("ol_aly"), wc))
    out.append(emit(runs_c, "gsis_id", "ol_run_success", wc * pl.col("ol_success"), wc))

    # WR/TE: team adjusted EPA on outside runs, scaled by the player's share of snaps
    outside = rr.filter(pl.col("run_gap") == "end").group_by("game_id", "team").agg(
        (pl.col("w") * pl.col("ol_epa")).sum().alias("v"), pl.col("w").sum().alias("n"))
    sk = pg.join(pos.filter(pl.col("position").is_in(["WR", "TE"])), on="gsis_id").filter(pl.col("off_snaps") > 0) \
        .join(outside, on=["game_id", "team"], how="inner") \
        .with_columns((pl.col("off_snaps") / pl.col("team_off_snaps")).clip(0, 1).alias("share"))
    out.append(emit(sk, "gsis_id", "skill_run_block", pl.col("v") * pl.col("share"), pl.col("n") * pl.col("share")))
    return concat(out)


def g_size(group: list[str]) -> float:
    """Rates are per pass-block snap of the unit: 2 tackles, 3 interior linemen."""
    return 2.0 if group == ["OT"] else 3.0
