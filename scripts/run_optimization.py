#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import yaml
import os
import sys
import json
import time
import logging
from logging.handlers import RotatingFileHandler

from components.prompt_optimizer import PromptOptimizer

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def main():
    # -------- logging setup --------
    LOG_PATH = os.environ.get("OPT_LOG", "opt_8000.log")
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(LOG_PATH, maxBytes=10_000_000, backupCount=5, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("opt")

    # -------- CLI --------
    ap = argparse.ArgumentParser(description="Run prompt optimization")
    ap.add_argument("--mode", required=True, choices=["spo", "opro", "entropy", "opro_entropy"],
                    help="optimization mode")
    ap.add_argument("--config", default="config/models.yaml",
                    help="models + endpoints config yaml")
    ap.add_argument("--template", default="settings/sample_task.yaml",
                    help="task template yaml (single prompt)")
    ap.add_argument("--name", required=True,
                    help="workspace project name")
    ap.add_argument("--subset", default=None,
                    help="optional subset label (e.g., writing/roleplay/...)")
    ap.add_argument("--tag", default=None,
                    help="optional free-form tag for this run (e.g., OPRO, OPRO+ENT, baseline)")
    args = ap.parse_args()

    # -------- config & template sanity --------
    try:
        cfg = load_yaml(args.config)
    except Exception as e:
        raise RuntimeError(f"Failed to load config yaml: {args.config}") from e

    if not isinstance(cfg, dict) or not all(k in cfg for k in ("optimize_model", "evaluate_model", "execute_model")):
        raise RuntimeError(f"Bad config file (missing optimize/evaluate/execute blocks): {args.config}")

    if not os.path.isfile(args.template):
        raise FileNotFoundError(f"Template not found: {args.template}")

    # -------- workspace pre-create & top-level meta --------
    ws_project_root = ensure_dir(os.path.join("workspace", args.name))
    top_meta_path = os.path.join(ws_project_root, "meta.json")
    try:
        top_meta = {
            "project": args.name,
            "template": args.template,
            "mode": args.mode,
            "subset": args.subset,
            "tag": args.tag,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(top_meta_path, "w", encoding="utf-8") as f:
            json.dump(top_meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to write top-level meta.json (%s): %s", top_meta_path, e)

    # -------- construct optimizer --------
    logger.info("Starting optimization | mode=%s | template=%s | name=%s | subset=%s | tag=%s",
                args.mode, args.template, args.name, args.subset, args.tag)

    po = PromptOptimizer(cfg, args.mode, args.template, args.name)
    try:
        prompts_dir = ensure_dir(os.path.join("workspace", args.name, "prompts"))
        cli_meta_path = os.path.join(prompts_dir, "cli_meta.json")
        cli_meta = {
            "subset": args.subset,
            "tag": args.tag,
            "note": "CLI-provided labels; kept separate to avoid overwriting prompts/meta.json"
        }
        with open(cli_meta_path, "w", encoding="utf-8") as f:
            json.dump(cli_meta, f, ensure_ascii=False, indent=2)
        logger.info("CLI meta written -> %s", cli_meta_path)
    except Exception as e:
        logger.warning("Failed to write CLI meta into prompts/: %s", e)

    # -------- run --------
    results_json = po.optimize()
    logger.info("Finished. Results -> %s", results_json)

if __name__ == "__main__":
    main()
