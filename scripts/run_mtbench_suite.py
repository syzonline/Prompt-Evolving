#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Original/SPO/OPRO/Ours on MT-Bench in one command."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Dict


MODE_MAP: Dict[str, str] = {
    "original": "none",
    "spo": "spo",
    "opro": "opro",
    "ours": "current",
}


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One-click MT-Bench suite runner.")
    ap.add_argument("--config", default="config/models.yaml", help="Model endpoint config YAML.")
    ap.add_argument("--template", required=True, help="Optimizer template YAML (requirements/sampling).")
    ap.add_argument("--mtbench-dir", default="data/mt-bench", help="MT-Bench local cache/download path.")
    ap.add_argument("--mtbench-split", default="train", help="Dataset split, e.g. train.")
    ap.add_argument("--mtbench-categories", default="", help="Optional categories, comma-separated.")
    ap.add_argument("--mtbench-limit", type=int, default=None, help="Optional sample cap for quick tests.")
    ap.add_argument("--output-root", default="workspace/mtbench_suite", help="Suite output root.")
    ap.add_argument("--project-prefix", default="mtbench", help="Project name prefix.")
    ap.add_argument("--modes", default="original,spo,opro,ours",
                    help="Comma-separated: original,spo,opro,ours")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


def main() -> int:
    args = build_parser().parse_args()
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    selected = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    for key in selected:
        if key not in MODE_MAP:
            raise ValueError(f"Unsupported mode alias: {key}. Use one of {list(MODE_MAP)}")

    root = Path(args.output_root) / ts
    root.mkdir(parents=True, exist_ok=True)

    for key in selected:
        mode = MODE_MAP[key]
        project = f"{args.project_prefix}_{key}_{ts}"
        out_dir = root / key
        log_file = out_dir / "run.log"
        cmd = [
            "python", "-m", "scripts.run_optimizer",
            "--mode", mode,
            "--dataset", "mt-bench",
            "--template", args.template,
            "--project", project,
            "--config", args.config,
            "--mtbench-dir", args.mtbench_dir,
            "--mtbench-split", args.mtbench_split,
            "--output-dir", str(out_dir),
            "--log-file", str(log_file),
            "--log-level", args.log_level,
        ]
        if args.mtbench_categories:
            cmd.extend(["--mtbench-categories", args.mtbench_categories])
        if args.mtbench_limit is not None:
            cmd.extend(["--mtbench-limit", str(args.mtbench_limit)])

        run(cmd)

    print(f"[DONE] suite outputs -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
