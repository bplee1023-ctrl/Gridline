"""Build the modeling baseline from history (run once per season, or when methodology changes).

    python -m pipeline.baseline.build            # writes data/baseline/
    python -m pipeline.baseline.build --upload   # also uploads to the repo's `baseline` release

Outputs
  baseline_stats.json        everything the weekly grader needs (small)
  player_games_<season>.parquet  per-player-game metric ledger for 2022-2025 (for refits/backtests)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import polars as pl
from scipy.optimize import nnls

from .. import ingest
from ..config import BASELINE_DIR, BASELINE_SEASONS, BOARDS, DEFENSE, MODEL_SEASONS, weights_cfg
from ..grade import (PosModel, board_players, exposure, fit_calibration, matrices, qualification, score)
from ..models import shrink
from ..positions import COMPONENTS, METRICS
from ..positions import k as kmod
from ..positions import p as pmod
from ..season import SeasonData, compute

log = logging.getLogger("gridline")

TARGET = {
    "QB": ["qb_epa_db", "qb_designed_run_epa", "qb_scramble_epa"],
    "RB": ["rush_epa", "rec_epa_tgt"], "WR": ["rec_epa_tgt"], "TE": ["rec_epa_tgt"],
    "OT": ["_team_off_epa"], "IOL": ["_team_off_epa"],
    "EDGE": ["_team_def_epa"], "IDL": ["_team_def_epa"], "LB": ["_team_def_epa"], "CB": ["_team_def_epa"],
    "S": ["_team_def_epa"],
    "K": ["fg_poe", "xp_oe", "ko_epa_oe"], "P": ["punt_ep_oe"], "RET": ["ret_epa_oe"],
}
TEAM_TARGET = {"OT", "IOL"} | DEFENSE


def fit_models() -> dict:
    log.info("fitting kicking/punting models on %s-%s", MODEL_SEASONS[0], MODEL_SEASONS[-1])
    pbp = ingest.load_pbp_only(MODEL_SEASONS)
    m = kmod.models_from(pbp)
    m.update(pmod.models_from(pbp))
    return m


def target_value(pos: str, long: pl.DataFrame, q: pl.DataFrame, ids: list[str]) -> np.ndarray:
    t = long.filter(pl.col("metric").is_in(TARGET[pos]) & pl.col("gsis_id").is_in(ids)) \
        .group_by("gsis_id").agg(pl.col("num").sum(), pl.col("den").sum())
    e = q.select("gsis_id", exposure(pos, q).alias("expo"))
    t = pl.DataFrame({"gsis_id": ids}).join(t, on="gsis_id", how="left").join(e, on="gsis_id", how="left")
    if pos in TEAM_TARGET:
        v = t["num"] / t["den"]
    else:
        v = t["num"] / t["expo"]
    return v.fill_nan(None).to_numpy().astype(float)


def fit_weights(Z: np.ndarray, y: np.ndarray, prior: np.ndarray, strength: float) -> tuple[np.ndarray, float]:
    ok = ~np.isnan(y)
    Z, y = np.nan_to_num(Z[ok]), y[ok]
    if len(y) < 20:
        return prior, float("nan")
    y = y - y.mean()
    Zp = Z @ prior
    s = float(Zp @ y / max(Zp @ Zp, 1e-12))
    s = max(s, 0.0)
    lam = strength * len(y) * float(np.mean(np.var(Z, axis=0)) or 1.0)
    A = np.vstack([Z, np.sqrt(lam) * np.eye(len(prior))])
    b = np.concatenate([y, np.sqrt(lam) * s * prior])
    w, _ = nnls(A, b)
    pred = Z @ w
    r2 = 1 - float(((y - pred) ** 2).sum() / max((y ** 2).sum(), 1e-12))
    if w.sum() <= 0:
        return prior, r2
    return w / w.sum(), r2


def season_qual(sd: SeasonData, pos: str, ids: list[str]) -> pl.DataFrame:
    tg = sd.ctx.team_games
    return qualification(pos, sd.counts, tg, ids)


def build(upload: bool = False) -> dict:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = weights_cfg()
    t0 = time.time()
    models = fit_models()
    seasons: dict[int, SeasonData] = {}
    for s in BASELINE_SEASONS:
        log.info("computing metrics for %s", s)
        sd = compute(ingest.load_season(s), models)
        seasons[s] = sd
        sd.long.join(sd.positions, on="gsis_id", how="left").write_parquet(BASELINE_DIR / f"player_games_{s}.parquet")
        sd.counts.write_parquet(BASELINE_DIR / f"counts_{s}.parquet")

    stats: dict = {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "seasons": BASELINE_SEASONS, "model_seasons": [MODEL_SEASONS[0], MODEL_SEASONS[-1]],
                   "models": models, "metrics": {}, "weights": {}, "calibration": {}, "components": {},
                   "composite_sd": {}, "replacement": {}, "impact": {}}

    for pos in BOARDS:
        log.info("baseline stats: %s", pos)
        metrics = [m for c in COMPONENTS[pos] for m in COMPONENTS[pos][c]]
        per_season = {}
        for s, sd in seasons.items():
            ids = board_players(sd, pos, sd.counts)
            q = season_qual(sd, pos, ids)
            qset = set(q.filter(pl.col("meets_minimum"))["gsis_id"].to_list())
            per_season[s] = (ids, q, qset)

        # step 3: k from split-half reliability; position mean from qualified players
        ms: dict = {}
        for m in metrics:
            frames = []
            for s, sd in seasons.items():
                ids, _, qset = per_season[s]
                g = sd.long.filter((pl.col("metric") == m) & pl.col("gsis_id").is_in(ids)) \
                    .sort("week").with_columns(pl.col("game_id").rank("dense").over("gsis_id").alias("game_idx"),
                                               pl.lit(s).alias("season"))
                frames.append(g)
            pg = pl.concat(frames) if frames else pl.DataFrame()
            if pg.height == 0:
                ms[m] = {"k": 50.0, "mu": 0.0, "sd": 1.0, "n": 0, "sum": 0.0, "sumsq": 0.0}
                continue
            k = shrink.estimate_k(pg.select("season", "gsis_id", "game_idx", "num", "den"), METRICS[m].min_den_half)
            qual = pl.concat([pg.filter((pl.col("season") == s) & pl.col("gsis_id").is_in(list(per_season[s][2])))
                              for s in seasons])
            mu = float(qual["num"].sum() / qual["den"].sum()) if qual.height and qual["den"].sum() > 0 else \
                float(pg["num"].sum() / max(pg["den"].sum(), 1e-9))
            ms[m] = {"k": float(k if k is not None else 50.0), "k_estimated": k is not None, "mu": mu}
        stats["metrics"][pos] = ms

        # shrunk-rate distribution among qualified player-seasons -> sd, pool sums, quantiles
        bs_tmp = {"metrics": {pos: ms}}
        model = PosModel.from_stats(pos, bs_tmp, weights={"prior": cfg["priors"][pos]})
        allxh = []
        for s, sd in seasons.items():
            ids, q, qset = per_season[s]
            qids = [i for i in ids if i in qset]
            num, den = matrices(sd.long, qids, model.metrics)
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
                allxh.append((den * x + model.k * model.mu) / (den + model.k))
        X = np.vstack(allxh) if allxh else np.zeros((0, len(metrics)))
        for i, m in enumerate(model.metrics):
            v = X[:, i] if len(X) else np.array([])
            v = v[~np.isnan(v)]
            if len(v) > 1:
                ms[m].update(sd=float(v.std()) or 1.0, n=float(len(v)), sum=float(v.sum()), sumsq=float((v ** 2).sum()),
                             q=[float(x) for x in np.quantile(v, np.linspace(0, 1, 101))])
            else:
                ms[m].update(sd=1.0, n=0.0, sum=0.0, sumsq=0.0)

        # step 5: weights — early-season components predict later-season value
        prior = np.array([cfg["priors"][pos][c] for c in COMPONENTS[pos]], float)
        e0, e1 = cfg["fit"]["early_weeks"]
        l0, l1 = cfg["fit"]["later_weeks"]
        model = PosModel.from_stats(pos, {"metrics": {pos: ms}}, weights={"prior": cfg["priors"][pos]})
        Zs, ys = [], []
        for s, sd in seasons.items():
            ids, q, qset = per_season[s]
            early = sd.long.filter(pl.col("week").is_between(e0, e1))
            late = sd.long.filter(pl.col("week").is_between(l0, l1))
            late_counts = sd.counts.filter(pl.col("week").is_between(l0, l1))
            num, den = matrices(early, ids, model.metrics)
            sc = score(model, num, den)
            lq = qualification(pos, late_counts, sd.ctx.team_games.join(
                sd.ctx.games.filter(pl.col("week").is_between(l0, l1)).select("game_id"), on="game_id"), ids)
            y = target_value(pos, late, lq, ids)
            keep = np.array([i in qset for i in ids]) & ~np.isnan(sc["composite"]) & ~np.isnan(y)
            Zs.append(sc["cz"][keep])
            ys.append(y[keep])
        Z = np.vstack(Zs) if Zs else np.zeros((0, len(prior)))
        y = np.concatenate(ys) if ys else np.zeros(0)
        pn = prior / prior.sum()
        w, r2 = fit_weights(Z, y, pn, cfg["fit"]["prior_strength"])
        # trust the fit in proportion to how much future value it explains (at most half),
        # so a fit with no predictive signal falls back to the priors
        alpha = 0.0 if np.isnan(r2) or r2 <= 0 else float(min(0.5, r2 / 0.10))
        final = alpha * w + (1 - alpha) * pn
        stats["weights"][pos] = {
            "prior": dict(zip(COMPONENTS[pos], map(float, prior))),
            "fitted": dict(zip(COMPONENTS[pos], [round(float(x), 4) for x in w])),
            "final": dict(zip(COMPONENTS[pos], [round(float(x), 4) for x in final / final.sum()])),
            "fit_share": round(alpha, 3),
            "r2": None if np.isnan(r2) else round(r2, 4), "n": int(len(y)),
            "target": TARGET[pos],
        }

        # step 6 + Impact: full-season composites of qualified players
        model = PosModel.from_stats(pos, {"metrics": {pos: ms}}, weights=stats["weights"][pos])
        comps, czs, vals, repl = [], [], [], []
        for s, sd in seasons.items():
            ids, q, qset = per_season[s]
            qids = [i for i in ids if i in qset]
            num, den = matrices(sd.long, qids, model.metrics)
            sc = score(model, num, den)
            comps.append(sc["composite"])
            czs.append(sc["cz"])
            vals.append(target_value(pos, sd.long, q, qids))
            c = np.sort(sc["composite"][~np.isnan(sc["composite"])])[::-1]
            n = cfg["replacement_rank"][pos]
            if len(c):
                repl.append(float(c[min(n, len(c)) - 1]))
        comp = np.concatenate(comps) if comps else np.zeros(0)
        cz = np.vstack(czs) if czs else np.zeros((0, len(prior)))
        stats["calibration"][pos] = fit_calibration(comp)
        # game grades get their own scale: a 90 is an elite single game
        gcomps = []
        for s_, sd in seasons.items():
            ids, q, qset = per_season[s_]
            pgk = sd.long.filter(pl.col("gsis_id").is_in(list(qset)) & pl.col("metric").is_in(model.metrics)) \
                .select("gsis_id", "game_id").unique()
            keys = list(zip(pgk["gsis_id"].to_list(), pgk["game_id"].to_list()))
            gn, gd = matrices(sd.long, keys, model.metrics, keys=["gsis_id", "game_id"])
            gcomps.append(score(model, gn, gd, k_mult=1.25)["composite"])
        stats.setdefault("game_calibration", {})[pos] = fit_calibration(np.concatenate(gcomps) if gcomps else np.zeros(0))
        stats["composite_sd"][pos] = float(np.nanstd(comp)) if len(comp) else 1.0
        stats["components"][pos] = {c: {"sd": float(np.nanstd(cz[:, i])) if len(cz) else 1.0}
                                    for i, c in enumerate(COMPONENTS[pos])}
        stats["replacement"][pos] = float(np.mean(repl)) if repl else 0.0
        v = np.concatenate(vals) if vals else np.zeros(0)
        ok = np.isfinite(v) & np.isfinite(comp)
        beta, src = 0.0, "regression"
        if ok.sum() >= 20:
            beta = float(np.polyfit(comp[ok], v[ok], 1)[0])
        fb = cfg["impact_scale_fallback"].get(pos)
        if fb is not None:
            if beta <= 0:
                beta, src = fb, "fallback"
            elif not (0.25 * fb <= beta <= 3 * fb):
                beta, src = float(np.clip(beta, 0.25 * fb, 3 * fb)), "regression (capped)"
        elif beta <= 0:
            beta, src = 0.0, "unavailable"
        stats["impact"][pos] = {"beta": beta, "source": src,
                                "unit": "EP per snap per composite SD-unit" if pos not in {"K", "P", "RET"}
                                else "EP per attempt per composite unit"}

    from .. import validate
    log.info("running validation backtests")
    stats["validation"] = validate.run(seasons, stats, target_value)
    for pos, v in stats["validation"].items():
        log.info("%-4s stability r=%s (best metric %s)  predictive r=%s vs %s", pos, v["stability"]["grade_r"],
                 v["stability"]["best_single_metric_r"], v["predictiveness"]["grade_r"], v["predictiveness"]["comparators"])

    path = BASELINE_DIR / "baseline_stats.json"
    path.write_text(json.dumps(stats, separators=(",", ":")))
    log.info("baseline written to %s in %.0fs", path, time.time() - t0)
    if upload:
        files = [str(path)] + [str(p) for p in BASELINE_DIR.glob("*.parquet")]
        subprocess.run(["gh", "release", "create", "baseline", "--title", "Modeling baseline",
                        "--notes", "Built by pipeline.baseline.build. Rebuilt at season end."], check=False)
        subprocess.run(["gh", "release", "upload", "baseline", *files, "--clobber"], check=True)
    return stats


def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    build(upload=a.upload)


if __name__ == "__main__":
    main()
