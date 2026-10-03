"""Golden snapshot: a frozen 2025 week must reproduce its grades exactly."""
import json
from pathlib import Path

import pytest

from pipeline import ingest
from pipeline.config import BOARDS
from pipeline.grade import grade_board
from pipeline.season import compute

GOLD = Path(__file__).parent / "fixtures" / "golden"


def grade_frozen() -> dict:
    bs = json.loads((GOLD / "baseline_stats.json").read_text())
    sd = compute(ingest.load_season(2025, frozen_dir=GOLD), bs["models"])
    out = {}
    for pos in BOARDS:
        rows, _ = grade_board(pos, sd, bs, "ALL", n_boot=50, with_details=False)
        out[pos] = {r["gsis_id"]: [r["grade"], r["grade_band"], r["impact"], r["meets_minimum"]] for r in rows}
    return out


@pytest.mark.skipif(not (GOLD / "expected.json").exists(), reason="run tests/make_golden.py first")
def test_golden_snapshot_reproduces():
    expected = json.loads((GOLD / "expected.json").read_text())
    got = json.loads(json.dumps(grade_frozen()))
    for pos in BOARDS:
        assert got[pos].keys() == expected[pos].keys(), f"{pos}: player set changed"
        for pid, vals in expected[pos].items():
            g = got[pos][pid]
            # identical on the machine that froze it; allow one rounding step (0.1) for BLAS
            # differences between CPUs so CI on another runner doesn't flake
            assert g[3] == vals[3], f"{pos} {pid}: qualification changed"
            for a, b in zip(g[:3], vals[:3]):
                assert (a is None and b is None) or abs(a - b) <= 0.1 + 1e-9, f"{pos} {pid}: {g} != {vals}"
