"""Shared constants and config loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("GRIDLINE_DATA", ROOT / "data"))
SITE_DIR = Path(os.environ.get("GRIDLINE_SITE", ROOT / "site"))
BASELINE_DIR = Path(os.environ.get("GRIDLINE_BASELINE", DATA_DIR / "baseline"))

SEASON = int(os.environ.get("GRIDLINE_SEASON", 2026))
BASELINE_SEASONS = [2022, 2023, 2024, 2025]          # metric baseline (FTN starts 2022)
MODEL_SEASONS = list(range(2015, 2026))              # kicking / punting models

BOARDS = ["QB", "RB", "WR", "TE", "OT", "IOL", "EDGE", "IDL", "LB", "CB", "S", "K", "P", "RET"]
POSITIONS = [b for b in BOARDS if b != "RET"]
OFFENSE = {"QB", "RB", "WR", "TE", "OT", "IOL"}
DEFENSE = {"EDGE", "IDL", "LB", "CB", "S"}
SEASON_TYPES = ["REG", "POST", "ALL"]

BOARD_NAMES = {
    "QB": "Quarterback", "RB": "Running back", "WR": "Wide receiver", "TE": "Tight end",
    "OT": "Offensive tackle", "IOL": "Interior offensive line", "EDGE": "Edge defender",
    "IDL": "Interior defensive line", "LB": "Linebacker", "CB": "Cornerback", "S": "Safety",
    "K": "Kicker", "P": "Punter", "RET": "Returners",
}


@lru_cache
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f) or {}


def weights_cfg() -> dict:
    return load_yaml("weights.yaml")


def minimums_cfg() -> dict:
    return load_yaml("minimums.yaml")


def overrides_cfg() -> dict:
    return {str(k): str(v) for k, v in (load_yaml("position_overrides.yaml") or {}).items()}
