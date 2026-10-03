"""Metric catalog and component definitions per position (design doc, "Metric catalog").

Every metric is stored as a per-player-per-game (num, den) pair; its rate is sum(num)/sum(den)
over whatever games are selected. That one structure gives season grades, game grades, the
Regular/Playoffs/All switch, odd/even split-half reliability and the bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    label: str
    source: str
    unit: str = "rate"          # display hint: epa, pct, yds, rate, pp
    lower_better: bool = False
    status: str = "measured"    # or "estimated"
    min_den_half: float = 5.0   # minimum sample per half for split-half reliability
    note: str = ""


M = Metric
METRICS: dict[str, Metric] = {
    # --- quarterback
    "qb_epa_db": M("Adj. EPA per dropback", "pbp", "epa", min_den_half=40),
    "qb_success": M("Adj. success rate (dropbacks)", "pbp", "pct", min_den_half=40),
    "qb_air_epa_att": M("Air EPA per attempt", "pbp", "epa", min_den_half=40),
    "qb_cpoe": M("CPOE", "pbp", "pp", min_den_half=40),
    "qb_catchable": M("Catchable-ball rate", "FTN", "pct", min_den_half=40),
    "qb_bad_throw": M("Bad-throw rate", "PFR", "pct", lower_better=True, min_den_half=40),
    "qb_int_worthy": M("Interception-worthy rate", "FTN", "pct", lower_better=True, min_den_half=40),
    "qb_fumble_db": M("Fumbles per dropback", "pbp", "pct", lower_better=True, min_den_half=40),
    "qb_epa_progression": M("Adj. EPA on 2nd+ read throws", "pbp+FTN", "epa", min_den_half=10),
    "qb_fault_sack": M("QB-fault sack rate", "FTN", "pct", lower_better=True, min_den_half=40),
    "qb_epa_pressure_look": M("Adj. EPA vs blitz / 5+ rushers", "pbp+FTN", "epa", min_den_half=15),
    "qb_sack_oe_ttt": M("Sack rate over expected (given time to throw)", "pbp+NGS", "pct", lower_better=True, min_den_half=40),
    "qb_oop_epa": M("Adj. EPA out of pocket", "pbp+FTN", "epa", min_den_half=8),
    "qb_designed_run_epa": M("Adj. EPA per designed run", "pbp", "epa", min_den_half=5),
    "qb_scramble_epa": M("Adj. EPA per scramble", "pbp", "epa", min_den_half=5),
    "qb_scramble_success": M("Scramble success rate", "pbp", "pct", min_den_half=5),
    "qb_sneak_oe": M("Sneak conversion over expected", "pbp+FTN", "pct", min_den_half=2),
    # --- rushing (RB)
    "ryoe": M("Rush yards over expected / carry", "NGS (pbp fallback)", "yds", min_den_half=25),
    "rush_epa": M("Adj. rush EPA / carry (box-adjusted)", "pbp+FTN", "epa", min_den_half=25),
    "rush_success": M("Adj. rush success rate (box-adjusted)", "pbp+FTN", "pct", min_den_half=25),
    "yac_carry": M("Yards after contact / carry", "PFR", "yds", min_den_half=25),
    "broken_tackles_touch": M("Broken tackles / touch", "PFR", "pct", min_den_half=25),
    "fumbles_touch": M("Fumbles / touch (lost = 1, recovered = ½)", "pbp", "pct", lower_better=True, min_den_half=25),
    "short_yardage_oe": M("Short-yardage & goal-line conversion over expected", "pbp", "pct", min_den_half=3),
    # --- receiving (RB/WR/TE)
    "rec_epa_tgt": M("Adj. EPA per target", "pbp", "epa", min_den_half=15),
    "rec_epa_route": M("Adj. EPA per team pass snap", "pbp+snaps", "epa", min_den_half=60),
    "rec_success_tgt": M("Adj. success rate on targets", "pbp", "pct", min_den_half=15),
    "targets_route": M("Targets per pass snap", "pbp+snaps", "pct", min_den_half=60),
    "separation": M("Average separation", "NGS", "yds", min_den_half=10),
    "first_read_share": M("First-read targets per pass snap", "FTN", "pct", min_den_half=60),
    "cr_oe": M("Catch rate over expected", "pbp", "pct", min_den_half=15),
    "drop_rate": M("Drop rate (of catchable)", "FTN", "pct", lower_better=True, min_den_half=10),
    "contested_win": M("Contested catches won", "FTN", "pct", min_den_half=3),
    "created_rec": M("Created receptions / target", "FTN", "pct", min_den_half=15),
    "yacoe": M("YAC over expected / reception", "NGS (pbp fallback)", "yds", min_den_half=10),
    "broken_tackles_rec": M("Broken tackles / reception", "PFR", "pct", min_den_half=10),
    # --- estimated blocking
    "pp_press_oe": M("Pressures allowed over expected / pass-block snap (slot-charged)", "PFR+pbp+NGS+FTN", "pct", status="estimated", min_den_half=80),
    "pp_sack_oe": M("Sacks allowed over expected / pass-block snap (slot-charged)", "pbp+NGS+FTN", "pct", status="estimated", min_den_half=80),
    "ol_run_epa": M("Adj. run EPA through assigned gaps", "pbp", "epa", status="estimated", min_den_half=20),
    "ol_aly": M("Adj. line yards through assigned gaps", "pbp", "yds", status="estimated", min_den_half=20),
    "ol_run_success": M("Adj. run success through assigned gaps", "pbp", "pct", status="estimated", min_den_half=20),
    "skill_run_block": M("Team adj. EPA on outside runs while on field", "pbp+snaps", "epa", status="estimated", min_den_half=10),
    "offball_press_oe": M("Team blitz pressure over expected while on field", "PFR+FTN", "pct", status="estimated", min_den_half=40),
    "rb_blitz_sack_oe": M("Team sack rate over expected on blitzes while on field", "pbp+FTN", "pct", status="estimated", min_den_half=15),
    "pen_off": M("Penalty yards / 100 snaps", "pbp", "yds", lower_better=True, min_den_half=150),
    "pen_def": M("Penalty yards / 100 snaps", "pbp", "yds", lower_better=True, min_den_half=150),
    # --- defense
    "pressure_rate": M("Pressures / pass-rush snap (opp-adj.)", "PFR+snaps", "pct", min_den_half=60),
    "hurry_rate": M("Hurries / pass-rush snap", "PFR+snaps", "pct", min_den_half=60),
    "knockdown_rate": M("QB knockdowns / pass-rush snap", "PFR+snaps", "pct", min_den_half=60),
    "sack_rate": M("Sacks / pass-rush snap", "PFR+snaps", "pct", min_den_half=60),
    "pressure_to_sack": M("Pressure-to-sack rate", "PFR", "pct", min_den_half=5),
    "run_stops": M("Run stops / run snap", "pbp+snaps", "pct", min_den_half=40),
    "tackle_depth": M("Average depth of run tackle", "pbp", "yds", lower_better=True, min_den_half=5),
    "tfl": M("Tackles for loss / run snap", "pbp+snaps", "pct", min_den_half=40),
    "missed_tackle_rate": M("Missed tackle rate", "PFR", "pct", lower_better=True, min_den_half=8),
    "cov_yds_snap": M("Yards allowed / coverage snap (opp-adj.)", "PFR+snaps", "yds", lower_better=True, min_den_half=60),
    "cov_tgt_rate": M("Targets / coverage snap", "PFR+snaps", "pct", lower_better=True, min_den_half=60),
    "forced_inc_rate": M("Forced-incompletion rate / target", "PFR", "pct", min_den_half=8),
    "int_rate": M("Interceptions / target", "PFR", "pct", min_den_half=8),
    "yac_allowed": M("YAC allowed / catch", "PFR", "yds", lower_better=True, min_den_half=5),
    "td_allowed": M("TDs allowed / target", "PFR", "pct", lower_better=True, min_den_half=8),
    "playmaking_epa": M("EPA swing of breakups & picks / coverage snap", "pbp+snaps", "epa", min_den_half=60),
    "blitz_pressure": M("Pressures / blitz", "PFR", "pct", min_den_half=5),
    # --- kicking / punting / returns
    "fg_poe": M("FG points over expected / attempt", "pbp + in-house model", "pts", min_den_half=6),
    "xp_oe": M("Extra-point makes over expected", "pbp + in-house model", "pct", min_den_half=6),
    "ko_epa_oe": M("Kickoff EP vs expectation", "pbp", "epa", min_den_half=10),
    "ko_oob": M("Kickoffs out of bounds or short of landing zone", "pbp", "pct", lower_better=True, min_den_half=10),
    "punt_ep_oe": M("Expected points saved / punt", "pbp + in-house model", "epa", min_den_half=8),
    "punt_in20_oe": M("Inside-20 rate over expected", "pbp + in-house model", "pct", min_den_half=8),
    "punt_tb_oe": M("Touchback rate over expected", "pbp + in-house model", "pct", lower_better=True, min_den_half=8),
    "punt_fc_oe": M("Fair-catch rate over expected", "pbp + in-house model", "pct", min_den_half=8),
    "punt_ret_yds_oe": M("Return yards allowed over expected (½ credit, sign: saved)", "pbp + in-house model", "yds", min_den_half=8),
    "ret_epa_oe": M("Return EPA over expected / attempt", "pbp + in-house model", "epa", min_den_half=5),
    "explosive_ret": M("Explosive-return rate", "pbp", "pct", min_den_half=5),
    "ret_fumble": M("Fumbles / return", "pbp", "pct", lower_better=True, min_den_half=5),
}

C = dict
COMPONENTS: dict[str, dict[str, list[str]]] = {
    "QB": C(
        passing_value=["qb_epa_db", "qb_success", "qb_air_epa_att"],
        accuracy=["qb_cpoe", "qb_catchable", "qb_bad_throw"],
        decisions=["qb_int_worthy", "qb_fumble_db", "qb_epa_progression"],
        pocket=["qb_fault_sack", "qb_epa_pressure_look", "qb_sack_oe_ttt", "qb_oop_epa"],
        rushing=["qb_designed_run_epa", "qb_scramble_epa", "qb_scramble_success", "qb_sneak_oe"],
        penalties=["pen_off"],
    ),
    "RB": C(
        rushing_efficiency=["ryoe", "rush_epa", "rush_success"],
        contact=["yac_carry", "broken_tackles_touch"],
        receiving=["rec_epa_tgt", "targets_route", "yacoe", "drop_rate"],
        pass_protection=["rb_blitz_sack_oe", "offball_press_oe"],
        ball_security=["fumbles_touch"],
        short_yardage=["short_yardage_oe"],
    ),
    "WR": C(
        receiving_value=["rec_epa_tgt", "rec_epa_route", "rec_success_tgt"],
        earning_targets=["targets_route", "separation", "first_read_share"],
        catch_quality=["cr_oe", "drop_rate", "contested_win", "created_rec"],
        after_catch=["yacoe", "broken_tackles_rec"],
        run_blocking=["skill_run_block"],
        ball_security=["fumbles_touch", "pen_off"],
    ),
    "TE": C(
        receiving_value=["rec_epa_tgt", "rec_epa_route", "rec_success_tgt"],
        earning_targets=["targets_route", "separation", "first_read_share"],
        catch_quality=["cr_oe", "drop_rate", "contested_win", "created_rec"],
        after_catch=["yacoe", "broken_tackles_rec"],
        run_blocking=["skill_run_block"],
        pass_protection=["offball_press_oe"],
        ball_security=["fumbles_touch", "pen_off"],
    ),
    "OT": C(pass_protection=["pp_press_oe", "pp_sack_oe"],
            run_blocking=["ol_run_epa", "ol_aly", "ol_run_success"], penalties=["pen_off"]),
    "IOL": C(pass_protection=["pp_press_oe", "pp_sack_oe"],
             run_blocking=["ol_run_epa", "ol_aly", "ol_run_success"], penalties=["pen_off"]),
    "EDGE": C(pass_rush=["pressure_rate", "hurry_rate", "knockdown_rate", "sack_rate", "pressure_to_sack"],
              run_defense=["run_stops", "tackle_depth", "tfl"], tackling=["missed_tackle_rate"],
              penalties=["pen_def"]),
    "IDL": C(pass_rush=["pressure_rate", "hurry_rate", "knockdown_rate", "sack_rate", "pressure_to_sack"],
             run_defense=["run_stops", "tackle_depth", "tfl"], tackling=["missed_tackle_rate"],
             penalties=["pen_def"]),
    "LB": C(run_defense=["run_stops", "tackle_depth", "tfl"],
            coverage=["cov_yds_snap", "cov_tgt_rate", "forced_inc_rate", "int_rate", "playmaking_epa"],
            blitzing=["blitz_pressure"], tackling=["missed_tackle_rate"], penalties=["pen_def"]),
    "CB": C(coverage=["cov_yds_snap", "cov_tgt_rate", "forced_inc_rate", "int_rate", "yac_allowed",
                      "td_allowed", "playmaking_epa"],
            run_support=["run_stops", "tackle_depth"], tackling=["missed_tackle_rate"], penalties=["pen_def"]),
    "S": C(coverage=["cov_yds_snap", "cov_tgt_rate", "forced_inc_rate", "int_rate", "yac_allowed",
                     "td_allowed", "playmaking_epa"],
           run_support=["run_stops", "tackle_depth"], tackling=["missed_tackle_rate"], penalties=["pen_def"]),
    "K": C(field_goals=["fg_poe"], extra_points=["xp_oe"], kickoffs=["ko_epa_oe", "ko_oob"]),
    "P": C(net_value=["punt_ep_oe"], pinning=["punt_in20_oe", "punt_tb_oe"],
           return_suppression=["punt_fc_oe", "punt_ret_yds_oe"]),
    "RET": C(return_value=["ret_epa_oe"], explosive=["explosive_ret"], ball_security=["ret_fumble"]),
}

COMPONENT_LABELS = {
    "passing_value": "Passing value", "accuracy": "Accuracy", "decisions": "Decisions & ball security",
    "pocket": "Pocket & pressure", "rushing": "Rushing", "penalties": "Penalties",
    "rushing_efficiency": "Rushing efficiency", "contact": "Contact & elusiveness",
    "receiving": "Receiving", "pass_protection": "Pass protection", "ball_security": "Ball security",
    "short_yardage": "Short yardage & goal line", "receiving_value": "Receiving value",
    "earning_targets": "Earning targets", "catch_quality": "Catch quality",
    "after_catch": "After the catch", "run_blocking": "Run blocking", "pass_rush": "Pass rush",
    "run_defense": "Run defense", "tackling": "Tackling", "coverage": "Coverage",
    "blitzing": "Blitzing", "run_support": "Run support", "field_goals": "Field goals",
    "extra_points": "Extra points", "kickoffs": "Kickoffs", "net_value": "Net value",
    "pinning": "Pinning", "return_suppression": "Return suppression", "return_value": "Return value",
    "explosive": "Explosive returns",
}

# Metrics described in the design doc that turned out to be unavailable in free data.
# Per the implementation spec they are dropped, their component renormalized, and listed
# on the Methodology screen.
DROPPED = [
    {"position": "QB", "component": "decisions", "metric": "Aggressiveness vs. outcome (NGS)",
     "reason": "NGS aggressiveness is a weekly aggregate with no play-level outcome to pair it with."},
    {"position": "CB/S/LB", "component": "coverage", "metric": "Shadow-assignment adjustment",
     "reason": "Coverage assignments are not public; coverage is opponent-adjusted at the team level."},
    {"position": "All", "component": "penalties", "metric": "Position-specific foul lists",
     "reason": "All accepted fouls charged to the player are counted; nflverse has no reliable foul-type taxonomy beyond the free-text penalty_type."},
]


def component_status(pos: str, comp: str) -> str:
    st = {METRICS[m].status for m in COMPONENTS[pos][comp]}
    return "estimated" if st == {"estimated"} else ("mixed" if len(st) > 1 else "measured")


def all_metric_names() -> list[str]:
    return list(METRICS)
