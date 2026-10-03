"""Write the snapshot the app reads (design doc, "Snapshot data contract")."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .config import BOARD_NAMES, BOARDS, ROOT, SEASON_TYPES, weights_cfg
from .grade import games_payload, grade_board, PosModel
from .positions import COMPONENT_LABELS, COMPONENTS, DROPPED, METRICS, component_status
from .season import SeasonData

log = logging.getLogger("gridline")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_default))


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    raise TypeError(type(o))


def _clean(obj):
    """Replace NaN/inf with None so the JSON is strict."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def board_meta(pos: str, bs: dict) -> dict:
    m = PosModel.from_stats(pos, bs)
    w = dict(zip(m.comps, m.w.tolist()))
    return {
        "board": pos, "name": BOARD_NAMES[pos],
        "components": [{"key": c, "label": COMPONENT_LABELS.get(c, c), "status": component_status(pos, c),
                        "weight": round(w[c], 4)} for c in COMPONENTS[pos]],
    }


TARGET_LABELS = {"_team_off_epa": "team offensive EPA per play while on the field",
                 "_team_def_epa": "team defensive EPA prevented per play while on the field"}


def methodology(bs: dict, sd: SeasonData, feeds_ts: dict) -> dict:
    cfg = weights_cfg()
    boards = {}
    for pos in BOARDS:
        wt = bs.get("weights", {}).get(pos, {})
        boards[pos] = {
            "name": BOARD_NAMES[pos],
            "weights": {"prior": wt.get("prior") or cfg["priors"][pos], "fitted": wt.get("fitted"),
                        "used": wt.get("final") if cfg.get("use_fitted", True) and wt.get("final") else cfg["priors"][pos],
                        "r2": wt.get("r2"), "n": wt.get("n"), "fit_share": wt.get("fit_share"),
                        "target": [TARGET_LABELS.get(t) or METRICS[t].label for t in (wt.get("target") or [])]},
            "validation": bs.get("validation", {}).get(pos),
            "components": [{
                "key": c, "label": COMPONENT_LABELS.get(c, c), "status": component_status(pos, c),
                "metrics": [{"metric": m, "label": METRICS[m].label, "source": METRICS[m].source,
                             "status": METRICS[m].status, "lower_better": METRICS[m].lower_better,
                             "k": round(bs["metrics"].get(pos, {}).get(m, {}).get("k", 0), 1),
                             "league_avg": bs["metrics"].get(pos, {}).get(m, {}).get("mu")}
                            for m in COMPONENTS[pos][c]],
            } for c in COMPONENTS[pos]],
            "calibration": bs.get("calibration", {}).get(pos),
            "impact": bs.get("impact", {}).get(pos),
            "replacement_rank": cfg["replacement_rank"][pos],
        }
    return _clean({
        "boards": boards,
        "calibration_table": [
            {"range": "90–100", "label": "Elite", "target": "Top ~3%"},
            {"range": "80–89", "label": "High quality", "target": "Next ~12%"},
            {"range": "70–79", "label": "Above average starter", "target": "Next ~25%"},
            {"range": "60–69", "label": "Average", "target": "Next ~30%"},
            {"range": "Below 60", "label": "Replacement level", "target": "Bottom ~30%"},
        ],
        "dropped_metrics": DROPPED,
        "feeds": feeds_ts,
        "baseline": {"built_at": bs.get("built_at"), "seasons": bs.get("seasons"), "model_seasons": bs.get("model_seasons")},
        "garbage_time": cfg["garbage_time"],
        "notes": [
            "Bootstrap bands resample the player's games (500 draws) and show half the 5th–95th percentile range.",
            "Estimated components are apportioned from team results by snap share until participation data is released after the Super Bowl.",
            "Penalties count all accepted fouls charged to the player.",
        ],
        "attribution": ["Play-by-play: nflverse / nflfastR (CC-BY 4.0)", "FTN Data via nflverse (CC-BY-SA 4.0)",
                        "Next Gen Stats, Pro-Football-Reference advanced stats and snap counts via nflverse"],
    })


def write_snapshot(sd: SeasonData, bs: dict, feeds_ts: dict, out: Path, n_boot: int | None = None) -> dict:
    season_dir = out / str(sd.season)
    if season_dir.exists():
        shutil.rmtree(season_dir)
    season_dir.mkdir(parents=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    players: dict[str, dict] = {}
    boards_index = {}
    for st in SEASON_TYPES:
        for pos in BOARDS:
            rows, details = grade_board(pos, sd, bs, st, n_boot=n_boot)
            meta = board_meta(pos, bs)
            _dump(season_dir / "rankings" / st / f"{pos}.json",
                  _clean({**meta, "season": sd.season, "season_type": st, "generated_at": generated, "rows": rows}))
            boards_index.setdefault(st, {})[pos] = len(rows)
            for r in rows:
                p = players.setdefault(r["gsis_id"], {"gsis_id": r["gsis_id"], "name": r["name"], "boards": {}})
                p["boards"].setdefault(st, {})[pos] = {**r, **details.get(r["gsis_id"], {})}
            log.info("%s %s: %d players", st, pos, len(rows))

    for pos in BOARDS:
        payload = games_payload(pos, sd, bs)
        if payload:
            _dump(season_dir / "games" / f"{pos}.json", _clean(payload))

    info = {r["gsis_id"]: r for r in sd.ctx.players.iter_rows(named=True)}
    pos_of = dict(zip(sd.positions["gsis_id"].to_list(), sd.positions["position"].to_list()))
    search = []
    for pid, p in players.items():
        i = info.get(pid, {})
        any_st = p["boards"].get("ALL") or next(iter(p["boards"].values()))
        team = next((b.get("team") for b in any_st.values() if b.get("team")), i.get("latest_team"))
        p.update(team=team, position=pos_of.get(pid) or next(iter(any_st)), headshot=i.get("headshot"))
        _dump(season_dir / "players" / f"{pid}.json", _clean(p))
        search.append({"id": pid, "name": p["name"], "team": team, "pos": p["position"],
                       "boards": sorted({b for st in p["boards"].values() for b in st})})
    search.sort(key=lambda s: s["name"])
    _dump(season_dir / "search_index.json", search)
    _dump(season_dir / "methodology.json", methodology(bs, sd, feeds_ts))

    files = {}
    for f in sorted(season_dir.rglob("*.json")):
        rel = f.relative_to(season_dir).as_posix()
        files[rel] = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    through = int(sd.ctx.games["week"].max() or 0)
    manifest = {
        "season": sd.season, "generated_at": generated, "through_week": through,
        "feeds": feeds_ts, "pending_games": sd.ctx.pending_games,
        "boards": boards_index, "files": files,
    }
    _dump(season_dir / "manifest.json", manifest)

    # web fallback: the same single-file UI, served next to the data
    ui = ROOT / "app" / "ui" / "index.html"
    if ui.exists():
        shutil.copy(ui, out / "index.html")
    (out / ".nojekyll").write_text("")
    _dump(out / "latest.json", {"season": sd.season, "manifest": f"{sd.season}/manifest.json"})
    return manifest
