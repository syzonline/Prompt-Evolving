#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, os, subprocess, time, yaml, shutil
from pathlib import Path

def subset_from_path(path: str) -> str:
    p = Path(path)
    return p.parent.name

def short_name(model_cfg_path: str) -> str:
    try:
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        full = cfg.get("execute_model", {}).get("name", "model")
        return Path(full).name.replace(":", "_").replace("/", "_")
    except Exception:
        return "model"

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. mt-bench/writing/*.yaml")
    ap.add_argument("--config", required=True, help="config/models.yaml")
    ap.add_argument("--name-prefix", default="MTBench")
    ap.add_argument("--modes", default="eval_only,origq,opro,spo,opro_entropy,entropy",
                    help="comma list of strategies")
    ap.add_argument("--sleep", type=float, default=0.0, help="optional gap between runs (seconds)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("[WARN] no files matched", args.glob); return

    mdl_short = short_name(args.config)
    tstamp = time.strftime("%m%d-%H%M%S")

    for yml in files:
        base = Path(yml).stem
        subset = subset_from_path(yml)
        for mode in [s.strip() for s in args.modes.split(",") if s.strip()]:
            proj = f"{args.name_prefix}_{subset}_{base}_{mode}_{mdl_short}_{tstamp}"
            if mode == "eval_only":
                run(["python","-m","scripts.run_eval_only",
                     "--config", args.config,
                     "--template", yml,
                     "--name", proj, "--subset", subset, "--tag", "IO"])
            elif mode == "origq":
                run(["python","-m","scripts.run_eval_only",
                     "--config", args.config,
                     "--template", yml,
                     "--name", proj, "--subset", subset, "--tag", "OrigQ",
                     "--prompt-empty"])
            elif mode in ("opro","spo","opro_entropy","entropy"):
                run(["python","-m","scripts.run_optimization",
                     "--mode", mode, "--config", args.config,
                     "--template", yml, "--name", proj,
                     "--subset", subset, "--tag", mode])
            else:
                print(f"[SKIP] unknown mode {mode}")
            if args.sleep > 0:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
