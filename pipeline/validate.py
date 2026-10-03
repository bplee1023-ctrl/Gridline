"""Backtests on 2022-2025 (design doc, "Validation and testing").

Stability       odd-game vs even-game grades, compared with every single input metric
Predictiveness  week 1-8 grade vs value in weeks 9-18, compared with raw counting stats
                (and passer rating for QBs)
Face validity   the top of each 2025 board is written out for manual review against All-Pro
                voting and public season grades; gaps get explained, never tuned away
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import BOARDS
from .grade import PosModel, board_players, matrices, qualification, score


def _corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 15 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)


def _raw_counting(pos: str, plays: pl.DataFrame, ngs_pass: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Comparators the grade must beat: per-player totals in the selected plays."""
    sc = plays.filter(pl.col("play_type").is_in(["pass", "run"]))
    out = {}
    if pos == "QB":
        out["passing yards"] = sc.group_by(pl.col("passer_player_id").alias("gsis_id")).agg(pl.col("yards_gained").sum().alias("v"))
        if ngs_pass.height:
            out["passer rating"] = ngs_pass.group_by("gsis_id").agg(
                ((pl.col("passer_rating") * pl.col("attempts")).sum() / pl.col("attempts").sum()).alias("v"))
    elif pos == "RB":
        out["rushing yards"] = sc.group_by(pl.col("rusher_player_id").alias("gsis_id")).agg(pl.col("yards_gained").sum().alias("v"))
    elif pos in ("WR", "TE"):
        out["receiving yards"] = sc.filter(pl.col("complete_pass") == 1).group_by(
            pl.col("receiver_player_id").alias("gsis_id")).agg(pl.col("yards_gained").sum().alias("v"))
    elif pos in ("EDGE", "IDL", "LB", "CB", "S"):
        cols = ["solo_tackle_1_player_id", "assist_tackle_1_player_id", "assist_tackle_2_player_id", "tackle_with_assist_1_player_id"]
        t = pl.concat([plays.select(pl.col(c).alias("gsis_id")) for c in cols]).drop_nulls()
        out["raw tackles"] = t.group_by("gsis_id").len("v")
        if pos in ("EDGE", "IDL"):
            out["sacks"] = plays.filter(pl.col("sack_player_id").is_not_null()).group_by(
                pl.col("sack_player_id").alias("gsis_id")).len("v")
    elif pos == "K":
        out["field goals made"] = plays.filter(pl.col("field_goal_result") == "made").group_by(
            pl.col("kicker_player_id").alias("gsis_id")).len("v")
    elif pos == "P":
        out["gross punt yards"] = plays.filter(pl.col("punt_attempt") == 1).group_by(
            pl.col("punter_player_id").alias("gsis_id")).agg(pl.col("kick_distance").mean().alias("v"))
    elif pos == "RET":
        out["return yards"] = pl.concat([
            plays.select(pl.col("punt_returner_player_id").alias("gsis_id"), "return_yards"),
            plays.select(pl.col("kickoff_returner_player_id").alias("gsis_id"), "return_yards")]) \
            .drop_nulls("gsis_id").group_by("gsis_id").agg(pl.col("return_yards").sum().alias("v"))
    return out


def run(seasons: dict, stats: dict, target_value) -> dict:
    report = {}
    for pos in BOARDS:
        model = PosModel.from_stats(pos, stats)
        stab_grade, stab_metric = [], {m: [] for m in model.metrics}
        pred_grade, pred_target, comparators = [], [], {}
        top = []
        for s, sd in seasons.items():
            ids = board_players(sd, pos, sd.counts)
            q = qualification(pos, sd.counts, sd.ctx.team_games, ids)
            qids = q.filter(pl.col("meets_minimum"))["gsis_id"].to_list()
            if not qids:
                continue
            # stability: odd vs even games
            g = sd.long.filter(pl.col("gsis_id").is_in(qids)).with_columns(
                (pl.col("game_id").rank("dense").over("gsis_id") % 2).alias("half"))
            n0, d0 = matrices(g.filter(pl.col("half") == 0), qids, model.metrics)
            n1, d1 = matrices(g.filter(pl.col("half") == 1), qids, model.metrics)
            s0, s1 = score(model, n0, d0), score(model, n1, d1)
            stab_grade.append((s0["grade"], s1["grade"]))
            for i, m in enumerate(model.metrics):
                stab_metric[m].append((s0["xh"][:, i], s1["xh"][:, i]))
            # predictiveness: weeks 1-8 -> weeks 9-18
            early = sd.long.filter(pl.col("week") <= 8)
            late = sd.long.filter(pl.col("week").is_between(9, 18))
            lc = sd.counts.filter(pl.col("week").is_between(9, 18))
            ltg = sd.ctx.team_games.join(sd.ctx.games.filter(pl.col("week").is_between(9, 18)).select("game_id"), on="game_id")
            ne, de = matrices(early, qids, model.metrics)
            se = score(model, ne, de)
            lq = qualification(pos, lc, ltg, qids)
            y = target_value(pos, late, lq, qids)
            pred_grade.append(se["composite"])
            pred_target.append(y)
            eplays = sd.ctx.plays.filter(pl.col("week") <= 8)
            engs = sd.ctx.ngs_pass.filter(pl.col("week") <= 8) if sd.ctx.ngs_pass.height else sd.ctx.ngs_pass
            for name, df in _raw_counting(pos, eplays, engs).items():
                d = dict(zip(df["gsis_id"].to_list(), df["v"].to_list()))
                comparators.setdefault(name, []).append(np.array([d.get(i, 0) or 0 for i in qids], float))
            if s == max(seasons):
                full = score(model, *matrices(sd.long, qids, model.metrics))
                names = dict(zip(sd.ctx.players["gsis_id"].to_list(), sd.ctx.players["name"].to_list()))
                order = np.argsort(-np.nan_to_num(full["grade"], nan=-1))[:10]
                top = [{"name": names.get(qids[i], qids[i]), "grade": round(float(full["grade"][i]), 1)} for i in order]
        if not stab_grade:
            continue
        a = np.concatenate([x[0] for x in stab_grade]); b = np.concatenate([x[1] for x in stab_grade])
        metric_r = {m: _corr(np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])) for m, v in stab_metric.items() if v}
        best_metric = max((r for r in metric_r.values() if r is not None), default=None)
        pg, pt = np.concatenate(pred_grade), np.concatenate(pred_target)
        comp_r = {k: _corr(np.concatenate(v), pt) for k, v in comparators.items()}
        g_r = _corr(pg, pt)
        report[pos] = {
            "stability": {"grade_r": _corr(a, b), "best_single_metric_r": best_metric,
                          "pass": _corr(a, b) is not None and best_metric is not None and _corr(a, b) > best_metric},
            "predictiveness": {"grade_r": g_r, "comparators": comp_r,
                               "pass": g_r is not None and all(r is None or abs(g_r) > abs(r) for r in comp_r.values())},
            "top_2025": top,
        }
    return report
