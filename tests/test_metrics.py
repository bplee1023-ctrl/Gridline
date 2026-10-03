"""Unit tests on hand-built fixtures."""
import json
import math

import numpy as np
import polars as pl
import pytest

from pipeline.grade import PosModel, fit_calibration, score
from pipeline.ingest import ContractError, check_contract
from pipeline.models import fg_model
from pipeline.models.adjust import adjust
from pipeline.models.shrink import estimate_k, shrink
from pipeline.positions import COMPONENTS, METRICS
from pipeline.positions._util import emit, tacklers
from pipeline.positions.ol import GAP_CREDIT


def test_every_component_metric_is_defined():
    for pos, comps in COMPONENTS.items():
        for c, ms in comps.items():
            for m in ms:
                assert m in METRICS, f"{pos}.{c}: {m}"


def test_no_banned_inputs_in_grades():
    banned = ("passer rating", "completion %", "raw yards", "raw tackles")
    for m in METRICS.values():
        assert not any(b in m.label.lower() for b in banned)


def test_emit_counts_rows_with_literal_denominator():
    df = pl.DataFrame({"pid": ["a", "a", "b"], "game_id": ["g", "g", "g"], "v": [1.0, 0.0, 1.0]})
    out = emit(df, "pid", "m", pl.col("v"), pl.lit(1.0)).sort("gsis_id")
    assert out["num"].to_list() == [1.0, 1.0]
    assert out["den"].to_list() == [2.0, 1.0]


def test_shrink_moves_small_samples_toward_mean():
    assert shrink(1.0, 0, 50, 0.2) == pytest.approx(0.2)
    assert shrink(1.0, 50, 50, 0.2) == pytest.approx(0.6)
    assert shrink(1.0, 1e9, 50, 0.2) == pytest.approx(1.0)


def test_estimate_k_recovers_known_reliability():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(400):
        true = rng.normal(0, 1)
        for g in range(16):
            n = 20
            num = rng.normal(true, 4.0, n).sum()   # per-play noise sd 4, signal sd 1 -> k = 16
            rows.append({"season": 1, "gsis_id": str(p), "game_idx": g, "num": num, "den": float(n)})
    k = estimate_k(pl.DataFrame(rows), 5)
    assert 10 < k < 25


def test_adjust_preserves_league_level_and_removes_opponent():
    rng = np.random.default_rng(1)
    n = 4000
    opp = rng.choice(["A", "B"], n)
    y = np.where(opp == "A", 0.3, -0.3) + rng.normal(0, 0.1, n)
    df = pl.DataFrame({
        "v": y, "p": rng.choice(["x", "y", "z"], n), "o": opp, "down": 1, "ydstogo": 10, "yardline_100": 50,
        "score_differential": 0, "qtr": 1, "wind": 5, "is_dome": 0.0, "is_home": 1.0, "w": 1.0,
    })
    out = adjust(df, "v", "p", "o", alpha=1.0)
    adj = out["v_adj"].to_numpy()
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-6)
    assert abs(adj[opp == "A"].mean() - adj[opp == "B"].mean()) < 0.05


def test_score_combines_components_and_renormalizes_missing():
    m = PosModel("X", ["a", "b"], ["c1", "c2"], np.array([0, 1]), np.array([1.0, -1.0]), np.array([10.0, 10.0]),
                 np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]),
                 60.0, 10.0, np.array([1.0, 1.0]), 1.0, ["measured", "estimated"])
    s = score(m, np.array([[20.0, 0.0]]), np.array([[10.0, 0.0]]))   # metric b missing
    assert s["composite"][0] == pytest.approx(1.0)                   # only c1 counts
    assert s["grade"][0] == pytest.approx(70.0)
    s2 = score(m, np.array([[0.0, 10.0]]), np.array([[10.0, 10.0]]))  # b is lower-better
    assert s2["cz"][0, 1] < 0


def test_calibration_hits_targets():
    c = np.random.default_rng(2).normal(0, 1, 5000)
    cal = fit_calibration(c)
    g = cal["a"] + cal["b"] * c
    assert np.mean(g >= 90) == pytest.approx(0.03, abs=0.02)
    assert np.mean(g < 60) == pytest.approx(0.30, abs=0.04)


def test_fg_model_monotone_in_distance():
    m = {"intercept": 6.0, "coef": {"dist": -4.0, "dist2": -3.0, "dist3": 2.0, "wind": -0.2, "cold": -0.05,
                                    "turf": 0.0, "dome": 0.05, "altitude": 0.0}}
    df = pl.DataFrame({"kick_distance": [25, 40, 55], "roof": ["outdoors"] * 3, "wind": [5, 5, 5],
                       "temp": [60, 60, 60], "surface": ["grass"] * 3, "home_team": ["KC"] * 3})
    p = fg_model.predict(df, m).to_list()
    assert p[0] > p[1] > p[2]


def test_tacklers_credit_solo_and_assists():
    plays = pl.DataFrame({
        "game_id": ["g"], "play_id": [1.0], "solo_tackle_1_player_id": [None], "solo_tackle_2_player_id": [None],
        "tackle_with_assist_1_player_id": [None], "tackle_with_assist_2_player_id": [None],
        "assist_tackle_1_player_id": ["a"], "assist_tackle_2_player_id": ["b"],
        "tackle_for_loss_1_player_id": ["a"], "tackle_for_loss_2_player_id": [None],
    }, schema_overrides={c: pl.String for c in ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                                                 "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id",
                                                 "tackle_for_loss_2_player_id"]})
    t = tacklers(plays).sort("tackler")
    assert t["credit"].to_list() == [0.5, 0.5]
    assert t["tfl"].to_list() == [1.0, 0.0]


def test_gap_credit_sums_to_one():
    for k, v in GAP_CREDIT.items():
        assert sum(v.values()) == pytest.approx(1.0), k


def test_schema_contract_rejects_missing_column():
    with pytest.raises(ContractError):
        check_contract("snaps", pl.DataFrame({"game_id": ["x"]}))


def test_minimums_match_pfr_season_totals():
    import yaml
    from pipeline.config import CONFIG_DIR
    m = yaml.safe_load(open(CONFIG_DIR / "minimums.yaml"))
    assert math.ceil(m["QB"]["per_team_game"] * 17) == 238
    assert math.ceil(m["RB"]["per_team_game"] * 17) == 107
    assert math.ceil(m["WR"]["per_team_game"] * 17) == 32
    assert math.ceil(m["P"]["per_team_game"] * 17) == 43
    assert math.ceil(m["RET"]["per_team_game"] * 17) == 22
