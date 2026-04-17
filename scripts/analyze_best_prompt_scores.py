#!/usr/bin/env python3
"""Analyze pairwise win-rates from best_prompt_scores_{spo,opro,ours}.json.

Default behavior matches the paper note in README:
- compares `score_shaped`
- focuses on turn=2 (0-based turn index 1)
- reports MT-Bench subsets: Writing / Roleplay / Humanities / STEM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

ScoreKey = Tuple[str, int]  # (qid, turn)


def build_subset_map() -> Dict[str, str]:
    """MT-Bench qid->subset mapping for selected four subsets.

    MT-Bench question IDs 81..160 are arranged by category in contiguous blocks.
    We only keep four subsets requested in the README analysis section.
    """

    mapping: Dict[str, str] = {}
    for qid in range(81, 91):
        mapping[str(qid)] = "Writing"
    for qid in range(91, 101):
        mapping[str(qid)] = "Roleplay"
    for qid in range(141, 151):
        mapping[str(qid)] = "STEM"
    for qid in range(151, 161):
        mapping[str(qid)] = "Humanities"
    return mapping


def load_scores(path: Path, score_field: str, turn: int, valid_qids: Iterable[str]) -> Dict[ScoreKey, float]:
    valid = set(valid_qids)
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[ScoreKey, float] = {}
    for row in rows:
        qid = str(row["qid"])
        row_turn = int(row["turn"])
        if row_turn != turn or qid not in valid:
            continue
        out[(qid, row_turn)] = float(row[score_field])
    return out


def pairwise_win_rate(
    a_scores: Mapping[ScoreKey, float], b_scores: Mapping[ScoreKey, float], keys: Iterable[ScoreKey]
) -> Tuple[float, float, float]:
    keys = list(keys)
    if not keys:
        return 0.0, 0.0, 0.0

    win = tie = lose = 0
    for k in keys:
        a, b = a_scores[k], b_scores[k]
        if a > b:
            win += 1
        elif a < b:
            lose += 1
        else:
            tie += 1
    n = len(keys)
    return win / n, tie / n, lose / n


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise win-rate analysis for best_prompt_scores*.json")
    parser.add_argument("--spo", default="best_prompt_scores_spo.json", type=Path)
    parser.add_argument("--opro", default="best_prompt_scores_opro.json", type=Path)
    parser.add_argument("--ours", default="best_prompt_scores_ours.json", type=Path)
    parser.add_argument("--score-field", default="score_shaped", choices=["score_shaped", "score_raw"])
    parser.add_argument(
        "--turn",
        type=int,
        default=1,
        help="0-based turn index; turn=1 corresponds to the 2nd turn (turn=2).",
    )
    args = parser.parse_args()

    subset_of_qid = build_subset_map()
    methods = {
        "SPO": load_scores(args.spo, args.score_field, args.turn, subset_of_qid.keys()),
        "OPRO": load_scores(args.opro, args.score_field, args.turn, subset_of_qid.keys()),
        "Ours": load_scores(args.ours, args.score_field, args.turn, subset_of_qid.keys()),
    }

    pairs = [("Ours", "SPO"), ("Ours", "OPRO"), ("OPRO", "SPO")]
    subsets = ["Writing", "Roleplay", "Humanities", "STEM"]

    print(f"# Win-rate analysis ({args.score_field}, turn_index={args.turn})")
    print()
    print("| Subset | Ours vs SPO | Ours vs OPRO | OPRO vs SPO |")
    print("|---|---:|---:|---:|")

    for subset in subsets:
        keys = [k for k in methods["SPO"].keys() if subset_of_qid[k[0]] == subset]
        cells: List[str] = []
        for left, right in pairs:
            w, t, l = pairwise_win_rate(methods[left], methods[right], keys)
            cells.append(f"{fmt_pct(w)} / {fmt_pct(t)} / {fmt_pct(l)}")
        print(f"| {subset} | {cells[0]} | {cells[1]} | {cells[2]} |")

    print("\nLegend: each cell is Win / Tie / Lose.")


if __name__ == "__main__":
    main()
