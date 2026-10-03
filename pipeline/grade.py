"""Metrics -> components -> 0-100 grade, Impact and bootstrap band (design doc steps 3-7)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import BOARDS, DEFENSE, OFFENSE, minimums_cfg, weights_cfg
from .positions import COMPONENTS, METRICS, component_status
from .season import SeasonData


# --------------------------------------------------------------------------- model per position
@dataclass
class PosModel:
    pos: str
    metrics: list[str]
    comps: list[str]
    comp_of: np.ndarray
    sign: np.ndarray
    k: np.ndarray
    mu: np.ndarray          # shrink target (position mean)
    mu_z: np.ndarray        # z-score centre (pooled 2022-2026 qualified)
    sd_z: np.ndarray
    w: np.ndarray
    a: float
    b: float
    comp_sd: np.ndarray
    composite_sd: float
    status: list[str]

    @classmethod
    def from_stats(cls, pos: str, bs: dict, weights: dict | None = None, pooled: dict | None = None) -> "PosModel":
        comps = list(COMPONENTS[pos])
        metrics = [m for c in comps for m in COMPONENTS[pos][c]]
        comp_of = np.array([ci for ci, c in enumerate(comps) for _ in COMPONENTS[pos][c]])
        ms = bs["metrics"].get(pos, {})
        k = np.array([ms.get(m, {}).get("k", 50.0) for m in metrics], float)
        mu = np.array([ms.get(m, {}).get("mu", 0.0) for m in metrics], float)
        sd = np.array([ms.get(m, {}).get("sd", 1.0) or 1.0 for m in metrics], float)
        mu_z = mu.copy()
        if pooled:
            for i, m in enumerate(metrics):
                b = ms.get(m)
                c = pooled.get(m)
                if b and c and b.get("n", 0) + c["n"] > 1:
                    n = b["n"] + c["n"]
                    s1 = b["sum"] + c["sum"]
                    s2 = b["sumsq"] + c["sumsq"]
                    mu_z[i] = s1 / n
                    var = max(s2 / n - mu_z[i] ** 2, 1e-12)
                    sd[i] = math.sqrt(var)
        wcfg = weights or bs.get("weights", {}).get(pos, {})
        use = "final" if weights_cfg().get("use_fitted", True) and "final" in wcfg else "prior"
        wd = wcfg.get(use) or weights_cfg()["priors"][pos]
        w = np.array([wd.get(c, 0.0) for c in comps], float)
        cal = bs.get("calibration", {}).get(pos, {"a": 65.0, "b": 12.0})
        csd = bs.get("components", {}).get(pos, {})
        comp_sd = np.array([csd.get(c, {}).get("sd", 1.0) or 1.0 for c in comps], float)
        return cls(pos, metrics, comps, comp_of, np.array([-1.0 if METRICS[m].lower_better else 1.0 for m in metrics]),
                   k, mu, mu_z, np.where(sd > 0, sd, 1.0), w, float(cal["a"]), float(cal["b"]), comp_sd,
                   float(bs.get("composite_sd", {}).get(pos, 1.0) or 1.0), [component_status(pos, c) for c in comps])


def score(model: PosModel, num: np.ndarray, den: np.ndarray, k_mult: float = 1.0) -> dict:
    """Vectorized steps 3-6 for any number of rows (players, games, or bootstrap draws)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        has = den > 0
        x = np.where(has, num / np.where(has, den, 1), np.nan)
        k = model.k * k_mult
        xh = np.where(has, (den * x + k * model.mu) / (den + k), np.nan)
        z = (xh - model.mu_z) / model.sd_z * model.sign
        C = len(model.comps)
        cz = np.full((z.shape[0], C), np.nan)
        n_in = np.zeros((z.shape[0], C))
        for c in range(C):
            cols = model.comp_of == c
            zc = z[:, cols]
            cnt = np.sum(~np.isnan(zc), axis=1)
            n_in[:, c] = cnt
            cz[:, c] = np.where(cnt > 0, np.nansum(zc, axis=1) / np.maximum(cnt, 1), np.nan)
        avail = ~np.isnan(cz)
        wsum = (avail * model.w).sum(1)
        composite = np.where(wsum > 0, np.nansum(np.nan_to_num(cz) * model.w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
        grade = np.clip(model.a + model.b * composite, 0, 100)
    return {"x": x, "xh": xh, "z": z, "cz": cz, "n_in": n_in, "wsum": wsum, "composite": composite, "grade": grade}


def comp_scores(model: PosModel, cz: np.ndarray) -> np.ndarray:
    """Component z -> 0-100 on the same scale as the grade."""
    return np.clip(model.a + model.b * (cz / model.comp_sd) * model.composite_sd, 0, 100)


def fit_calibration(composites: np.ndarray) -> dict:
    targets = weights_cfg()["calibration"]
    c = composites[~np.isnan(composites)]
    if len(c) < 8:
        return {"a": 65.0, "b": 12.0}
    q = np.array([np.quantile(c, t[0]) for t in targets])
    g = np.array([t[1] for t in targets], float)
    b, a = np.polyfit(q, g, 1)
    return {"a": float(a), "b": float(max(b, 1e-3))}


# --------------------------------------------------------------------------- helpers
def matrices(long: pl.DataFrame, ids: list[str], metrics: list[str], keys: list[str] = ["gsis_id"]):
    """Pivot (keys, metric, num, den) to dense [rows, metrics] arrays in `ids`/`metrics` order."""
    idx = {m: i for i, m in enumerate(metrics)}
    row = {r: i for i, r in enumerate(ids)}
    num = np.zeros((len(ids), len(metrics)))
    den = np.zeros((len(ids), len(metrics)))
    if long.height == 0:
        return num, den
    agg = long.filter(pl.col("metric").is_in(metrics)).group_by(keys + ["metric"]).agg(pl.col("num").sum(), pl.col("den").sum())
    keycol = agg[keys[0]].to_list() if len(keys) == 1 else list(zip(*[agg[k].to_list() for k in keys]))
    for key, m, n, d in zip(keycol, agg["metric"].to_list(), agg["num"].to_list(), agg["den"].to_list()):
        r = row.get(key)
        if r is not None:
            num[r, idx[m]] = n
            den[r, idx[m]] = d
    return num, den


def board_players(sd: SeasonData, pos: str, counts: pl.DataFrame) -> list[str]:
    if pos == "RET":
        c = counts.group_by("gsis_id").agg((pl.col("punt_returns") + pl.col("kick_returns")).sum().alias("r"))
        return sorted(c.filter(pl.col("r") > 0)["gsis_id"].to_list())
    ids = sd.positions.filter(pl.col("position") == pos)["gsis_id"]
    active = counts.group_by("gsis_id").agg((pl.col("off_snaps") + pl.col("def_snaps") + pl.col("fg_attempts")
                                             + pl.col("punts") + pl.col("pass_attempts") + pl.col("rush_attempts")
                                             + pl.col("targets")).sum().alias("t")).filter(pl.col("t") > 0)
    return sorted(set(ids.to_list()) & set(active["gsis_id"].to_list()))


QUAL_STAT = {"pass_attempts": "pass attempts", "rush_attempts": "rush attempts", "receptions": "receptions",
             "fg_attempts": "FG attempts", "punts": "punts", "returns": "returns", "snap_share": "snaps"}


def qualification(pos: str, counts: pl.DataFrame, team_games: pl.DataFrame, ids: list[str],
                  custom_min: float | None = None) -> pl.DataFrame:
    """qualifier_value, required minimum and meets_minimum per player (NFL per-team-game rules)."""
    rule = minimums_cfg()[pos]
    c = counts.filter(pl.col("gsis_id").is_in(ids))
    latest = c.sort("week").group_by("gsis_id").agg(pl.col("team").drop_nulls().last().alias("team"))
    tg = team_games.group_by("team").agg(pl.len().alias("team_games"),
                                         pl.col("team_off_snaps").sum().alias("t_off"),
                                         pl.col("team_def_snaps").sum().alias("t_def"))
    agg = c.group_by("gsis_id").agg(
        pl.col("pass_attempts").sum(), pl.col("rush_attempts").sum(), pl.col("receptions").sum(),
        pl.col("fg_attempts").sum(), pl.col("punts").sum(),
        pl.col("punt_returns").sum(), pl.col("kick_returns").sum(), pl.col("off_snaps").sum(),
        pl.col("def_snaps").sum(), pl.col("st_snaps").sum(), pl.col("xp_attempts").sum(), pl.col("kickoffs").sum(),
        pl.col("targets").sum(), pl.col("game_id").n_unique().alias("games"),
    ).join(latest, on="gsis_id", how="left").join(tg, on="team", how="left").with_columns(
        pl.col("team_games").fill_null(pl.col("games")), pl.col("t_off").fill_null(0), pl.col("t_def").fill_null(0))
    stat = rule["stat"]
    if stat == "snap_share":
        side = "off_snaps" if rule["side"] == "offense" else "def_snaps"
        tside = "t_off" if rule["side"] == "offense" else "t_def"
        agg = agg.with_columns(pl.col(side).alias("qualifier_value"),
                               (rule["share"] * pl.col(tside)).ceil().alias("required"))
    else:
        if stat == "returns":
            val = pl.max_horizontal("punt_returns", "kick_returns")
        else:
            val = pl.col(stat)
        agg = agg.with_columns(val.alias("qualifier_value"),
                               (rule["per_team_game"] * pl.col("team_games")).ceil().alias("required"))
    if custom_min is not None:
        agg = agg.with_columns(pl.lit(float(custom_min)).alias("required"))
    return agg.with_columns((pl.col("qualifier_value") >= pl.col("required")).alias("meets_minimum"),
                            pl.lit(QUAL_STAT.get(stat, stat)).alias("qualifier_stat"))


def exposure(pos: str, q: pl.DataFrame) -> pl.Expr:
    if pos == "K":
        return pl.col("fg_attempts") + pl.col("xp_attempts") + pl.col("kickoffs")
    if pos == "P":
        return pl.col("punts")
    if pos == "RET":
        return pl.col("punt_returns") + pl.col("kick_returns")
    if pos in DEFENSE:
        return pl.col("def_snaps")
    return pl.col("off_snaps")


def snaps_expr(pos: str) -> pl.Expr:
    if pos in DEFENSE:
        return pl.col("def_snaps")
    if pos in OFFENSE:
        return pl.col("off_snaps")
    return pl.col("st_snaps")


def pooled_current(model: PosModel, num: np.ndarray, den: np.ndarray, qualified: np.ndarray) -> dict:
    """Sufficient stats of current-season qualified shrunk rates, for the 2022-2026 pool."""
    out = {}
    if qualified.sum() == 0:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        has = den > 0
        x = np.where(has, num / np.where(has, den, 1), np.nan)
        xh = (den * x + model.k * model.mu) / (den + model.k)
    for i, m in enumerate(model.metrics):
        v = xh[qualified, i]
        v = v[~np.isnan(v)]
        if len(v):
            out[m] = {"n": float(len(v)), "sum": float(v.sum()), "sumsq": float((v ** 2).sum())}
    return out


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def percentile(bs: dict, pos: str, metric: str, value: float, lower_better: bool) -> float | None:
    q = bs["metrics"].get(pos, {}).get(metric, {}).get("q")
    if not q or value is None or np.isnan(value):
        return None
    p = float(np.interp(value, q, np.linspace(0, 100, len(q))))
    return round(100 - p if lower_better else p, 1)


# --------------------------------------------------------------------------- one board
def grade_board(pos: str, sd: SeasonData, bs: dict, season_type: str, n_boot: int | None = None,
                with_details: bool = True) -> tuple[list[dict], dict]:
    long = sd.long if season_type == "ALL" else sd.long.filter(pl.col("season_type") == season_type)
    counts = sd.counts if season_type == "ALL" else sd.counts.filter(pl.col("season_type") == season_type)
    if long.height == 0:
        return [], {}
    ids = board_players(sd, pos, counts)
    if not ids:
        return [], {}
    base = PosModel.from_stats(pos, bs)
    num, den = matrices(long, ids, base.metrics)

    tg = sd.ctx.team_games.join(sd.ctx.games.select("game_id", "season_type"), on="game_id")
    if season_type != "ALL":
        tg = tg.filter(pl.col("season_type") == season_type)
    q = qualification(pos, counts, tg, ids)
    qd = {r["gsis_id"]: r for r in q.iter_rows(named=True)}
    qualified = np.array([bool(qd.get(i, {}).get("meets_minimum")) for i in ids])

    model = PosModel.from_stats(pos, bs, pooled=pooled_current(base, num, den, qualified))
    s = score(model, num, den)
    cscore = comp_scores(model, s["cz"])

    # ---- bootstrap over the player's games
    n_boot = n_boot if n_boot is not None else int(weights_cfg().get("bootstrap_samples", 500))
    pg_keys = long.filter(pl.col("gsis_id").is_in(ids) & pl.col("metric").is_in(model.metrics)) \
        .select("gsis_id", "game_id").unique().sort("gsis_id", "game_id")
    pg_list = list(zip(pg_keys["gsis_id"].to_list(), pg_keys["game_id"].to_list()))
    gnum, gden = matrices(long, pg_list, model.metrics, keys=["gsis_id", "game_id"])
    by_player: dict[str, list[int]] = {}
    for r, (pid, _) in enumerate(pg_list):
        by_player.setdefault(pid, []).append(r)
    band = np.full(len(ids), np.nan)
    if n_boot > 0:
        for i, pid in enumerate(ids):
            rows = by_player.get(pid, [])
            if len(rows) < 2 or np.isnan(s["grade"][i]):
                band[i] = np.nan if not rows else 12.0
                continue
            rng = np.random.default_rng(_seed(sd.season, season_type, pos, pid))
            W = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=n_boot).astype(float)
            g = score(model, W @ gnum[rows], W @ gden[rows])["grade"]
            g = g[~np.isnan(g)]
            if len(g):
                band[i] = (np.percentile(g, 95) - np.percentile(g, 5)) / 2

    # ---- game grades (heavier shrinkage)
    gs = score(model, gnum, gden, k_mult=1.25)
    gcal = bs.get("game_calibration", {}).get(pos)
    if gcal:
        gs["grade"] = np.clip(gcal["a"] + gcal["b"] * gs["composite"], 0, 100)
    games_meta = {r["game_id"]: r for r in sd.ctx.games.iter_rows(named=True)}
    pending = set(sd.ctx.pending_games)

    # ---- Impact: expected points above replacement over the season
    imp = bs.get("impact", {}).get(pos, {})
    beta = float(imp.get("beta", 0.0))
    repl = float(bs.get("replacement", {}).get(pos, np.nanpercentile(s["composite"][qualified], 10) if qualified.any() else 0.0))
    expo = q.with_columns(exposure(pos, q).alias("expo"), snaps_expr(pos).alias("snaps"))
    ed = {r["gsis_id"]: r for r in expo.iter_rows(named=True)}

    rows = []
    details = {}
    players = sd.ctx.players.filter(pl.col("gsis_id").is_in(ids))
    pinfo = {r["gsis_id"]: r for r in players.iter_rows(named=True)}
    for i, pid in enumerate(ids):
        if np.isnan(s["composite"][i]):
            continue
        info = pinfo.get(pid, {})
        e = ed.get(pid, {})
        my_games = [pg_list[r][1] for r in by_player.get(pid, [])]
        avail = ~np.isnan(s["cz"][i])
        wsum = float((model.w * avail).sum())
        measured = float((model.w * avail * np.array([st == "measured" for st in model.status])).sum()
                         + 0.5 * (model.w * avail * np.array([st == "mixed" for st in model.status])).sum())
        game_rows = []
        for r in by_player.get(pid, []):
            gid = pg_list[r][1]
            gm = games_meta.get(gid, {})
            if not np.isnan(gs["grade"][r]):
                game_rows.append({"game_id": gid, "week": gm.get("week"), "grade": round(float(gs["grade"][r]), 1),
                                  "provisional": gid in pending})
        game_rows.sort(key=lambda g: (g["week"] or 0, g["game_id"]))
        impact = beta * (float(s["composite"][i]) - repl) * float(e.get("expo") or 0)
        row = {
            "gsis_id": pid, "name": info.get("name") or pid, "team": e.get("team"), "position": pos,
            "grade": round(float(s["grade"][i]), 1),
            "grade_band": None if np.isnan(band[i]) else round(float(band[i]), 1),
            "impact": round(impact, 1),
            "components": {c: (None if np.isnan(cscore[i, ci]) else round(float(cscore[i, ci]), 1))
                           for ci, c in enumerate(model.comps)},
            "snaps": int(e.get("snaps") or 0),
            "games": int(e.get("games") or 0),
            "qualifier_value": float(e.get("qualifier_value") or 0),
            "qualifier_required": float(e.get("required") or 0),
            "qualifier_stat": e.get("qualifier_stat"),
            "meets_minimum": bool(e.get("meets_minimum")),
            "measured_share": round(measured / wsum, 2) if wsum else 0.0,
            "trend": [g["grade"] for g in game_rows[-5:]],
            "provisional": any(g in pending for g in my_games),
        }
        rows.append(row)
        if with_details:
            metrics_out = []
            for mi, m in enumerate(model.metrics):
                ci = model.comp_of[mi]
                zval = s["z"][i, mi]
                contrib = None
                if not np.isnan(zval) and s["n_in"][i, ci] > 0 and wsum > 0:
                    contrib = round(float(model.b * model.w[ci] / wsum * zval / s["n_in"][i, ci]), 2)
                meta = METRICS[m]
                raw = s["x"][i, mi]
                metrics_out.append({
                    "metric": m, "label": meta.label, "component": model.comps[ci], "source": meta.source,
                    "unit": meta.unit, "status": meta.status, "lower_better": meta.lower_better,
                    "raw": None if np.isnan(raw) else round(float(raw), 4),
                    "shrunk": None if np.isnan(s["xh"][i, mi]) else round(float(s["xh"][i, mi]), 4),
                    "percentile": percentile(bs, pos, m, s["xh"][i, mi], meta.lower_better),
                    "league_avg": round(float(model.mu[mi]), 4),
                    "sample": round(float(den[i, mi]), 1),
                    "grade_points": contrib,
                })
            details[pid] = {"metrics": metrics_out, "game_log": game_rows,
                            "composite": round(float(s["composite"][i]), 4)}

    # ranks among qualified players; Impact rank too
    for key, rk in (("grade", "rank"), ("impact", "impact_rank")):
        ordered = sorted([r for r in rows if r["meets_minimum"]], key=lambda r: -r[key])
        for n, r in enumerate(ordered, 1):
            r[rk] = n
        for r in rows:
            r.setdefault(rk, None)
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, -r["grade"]))
    return rows, details


def label(grade: float) -> str:
    return ("Elite" if grade >= 90 else "High quality" if grade >= 80 else "Above average starter" if grade >= 70
            else "Average" if grade >= 60 else "Replacement level")


# --------------------------------------------------------------------------- client-side regrade data
def games_payload(pos: str, sd: SeasonData, bs: dict) -> dict | None:
    """Per-player-game metric sums plus the board's model, so the app can re-grade any week
    range or custom minimum exactly as the pipeline does (minus the bootstrap band)."""
    ids = board_players(sd, pos, sd.counts)
    if not ids:
        return None
    base = PosModel.from_stats(pos, bs)
    num, den = matrices(sd.long, ids, base.metrics)
    q = qualification(pos, sd.counts, sd.ctx.team_games, ids)
    qset = set(q.filter(pl.col("meets_minimum"))["gsis_id"].to_list())
    model = PosModel.from_stats(pos, bs, pooled=pooled_current(base, num, den, np.array([i in qset for i in ids])))

    pg = sd.long.filter(pl.col("gsis_id").is_in(ids) & pl.col("metric").is_in(model.metrics)) \
        .select("gsis_id", "game_id").unique().sort("gsis_id", "game_id")
    keys = list(zip(pg["gsis_id"].to_list(), pg["game_id"].to_list()))
    gnum, gden = matrices(sd.long, keys, model.metrics, keys=["gsis_id", "game_id"])
    rule = minimums_cfg()[pos]
    c = sd.counts.with_columns(exposure(pos, sd.counts).alias("expo"), snaps_expr(pos).alias("snaps"),
                               (pl.max_horizontal("punt_returns", "kick_returns") if rule["stat"] == "returns"
                                else pl.col(rule["side"][:3] + "_snaps" if rule["stat"] == "snap_share" else rule["stat"])
                                ).alias("qv"))
    cd = {(r["gsis_id"], r["game_id"]): r for r in c.iter_rows(named=True)}
    games = {r["game_id"]: r for r in sd.ctx.games.iter_rows(named=True)}
    names = {r["gsis_id"]: r["name"] for r in sd.ctx.players.iter_rows(named=True)}
    players: dict = {}
    for r, (pid, gid) in enumerate(keys):
        cr = cd.get((pid, gid), {})
        g = games.get(gid, {})
        p = players.setdefault(pid, {"name": names.get(pid, pid), "g": []})
        p["g"].append([gid, g.get("week"), g.get("season_type"), cr.get("team"),
                       [round(float(x), 4) for x in gnum[r]], [round(float(x), 3) for x in gden[r]],
                       float(cr.get("qv") or 0), float(cr.get("expo") or 0), float(cr.get("snaps") or 0)])
    team_games = {}
    side = "team_def_snaps" if pos in DEFENSE else "team_off_snaps"
    for r in sd.ctx.team_games.join(sd.ctx.games.select("game_id", "week", "season_type"), on="game_id").iter_rows(named=True):
        team_games.setdefault(r["team"], []).append([r["week"], r["season_type"], float(r.get(side) or 0)])
    imp = bs.get("impact", {}).get(pos, {})
    return {
        "board": pos, "metrics": model.metrics, "comps": model.comps, "comp_of": model.comp_of.tolist(),
        "sign": model.sign.tolist(), "k": model.k.tolist(), "mu": model.mu.tolist(), "mu_z": model.mu_z.tolist(),
        "sd_z": model.sd_z.tolist(), "w": model.w.tolist(), "a": model.a, "b": model.b,
        "comp_sd": model.comp_sd.tolist(), "composite_sd": model.composite_sd, "status": model.status,
        "game_cal": bs.get("game_calibration", {}).get(pos, {"a": model.a, "b": model.b}),
        "beta": float(imp.get("beta", 0.0)), "replacement": float(bs.get("replacement", {}).get(pos, 0.0)),
        "rule": rule, "team_games": team_games, "players": players,
    }
