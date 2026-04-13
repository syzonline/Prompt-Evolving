# -*- coding: utf-8 -*-
"""
Run the Continuous-Dialogue Prompt Optimizer (round-serial, turn-aware).

This script matches the new PromptOptimizer signature:
    PromptOptimizer(config: Dict, exec_llm, eval_llm, workdir: str)

Responsibilities
- Load template YAML (qa + optimizer knobs) and models YAML (HTTP endpoints).
- Map --mode to strategy:
    spo           -> entropy disabled, SPO-like candidate hook (rule-based light paraphrases)
    entropy       -> entropy enabled, default candidates (or SPO-lite), score shaping on
    opro          -> entropy disabled, OPRO-like candidate hook (use seeds from run/global)
    opro_entropy  -> entropy enabled + OPRO-like candidate hook
- Build OpenAI-compatible HTTP clients for exec_llm / eval_llm.
- Call opt.optimize(qa) and write artifacts under workspace/<project>/.

Usage example:
  python -m scripts.run_optimizer \
    --mode entropy \
    --template writing-demo/demo.yaml \
    --project writing-demo \
    --config config/models.yaml \
    --log-level INFO
"""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import yaml
import requests
from pathlib import Path

# IMPORTANT: import the new optimizer you pasted earlier
from components.prompt_optimizer import PromptOptimizer
from utils.mtbench_data import load_or_download_mtbench, mtbench_rows_to_qa

# ----------------------------
# Utilities
# ----------------------------
def load_config_any(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# OpenAI-compatible Chat Client
# ----------------------------
class ChatClient:
    """
    Minimal OpenAI-compatible /v1/chat/completions HTTP client.

    It works with local vLLM servers (OpenAI-compatible mode) and similar providers.
    Docs: https://docs.vllm.ai/.../openai_compatible_server.html
    """

    def __init__(self, name: str, endpoint: str, model: str, params: Dict[str, Any]):
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.params = params or {}
        self.session = requests.Session()

    def chat_complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if self.endpoint.endswith("/v1"):
            url = f"{self.endpoint}/chat/completions"
        else:
            url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
        }
        # merge default params (temperature, top_p, max_tokens, etc.)
        payload.update(self.params)
        # allow overrides per call
        payload.update(kwargs or {})
        try:
            resp = self.session.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible shape
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            return str(content)
        except Exception as e:
            logging.exception("[ChatClient:%s] request failed: %s", self.name, e)
            return ""

def build_client(models_cfg: Dict[str, Any], role_key: str, fallback_name: str) -> ChatClient:
    block = models_cfg.get(role_key) or {}
    name = block.get("model") or block.get("name", fallback_name)
    endpoint = block.get("base_url") or block.get("endpoint", "http://127.0.0.1:8002")
    params = block.get("params", {}) or {}
    return ChatClient(name=name, endpoint=endpoint, model=name, params=params)

# ----------------------------
# Candidate Hooks (SPO / OPRO)
# ----------------------------
def spo_hook(qid: str, turn: int, base: str, N: int, seeds: Optional[List[Tuple[str, float]]] = None) -> List[str]:
    """
    SPO-like: generate light paraphrases that preserve semantics
    (no template/policy injection). Ignore seeds.
    """
    base = (base or "").strip()
    if not base:
        return [base]

    def norm(x: str) -> str:
        return " ".join(x.split())

    cands = {norm(base)}

    # very light paraphrases (structure/synonyms/hints to reference prior turns)
    if len(cands) < N:
        cands.add(norm(f"{base} Please answer directly and reference earlier turns if relevant."))
    if len(cands) < N:
        cands.add(norm(base.replace("Describe", "Detail").replace("Explain", "Clarify")))
    if len(cands) < N and len(base.split()) > 8:
        cands.add(norm(base + " Keep your wording concrete and specific."))

    return list(cands)[:N]

def opro_hook(qid: str, turn: int, base: str, N: int, seeds: Optional[List[Tuple[str, float]]] = None) -> List[str]:
    """
    OPRO-like: reuse best recent rewrites (seeds) for THIS turn, mutate slightly.
    We DO NOT add generic slogans; only minimal clarifying edits.
    """
    base = (base or "").strip()
    pool: List[str] = [base] if base else []

    seeds = seeds or []
    # take top-2 seeds by score (if scores are available), else latest few
    seeds_sorted = sorted(seeds, key=lambda x: x[1], reverse=True)[:2] if seeds else []
    for s, _score in seeds_sorted:
        if s and s not in pool:
            pool.append(s)

    # mutate minimally
    out: List[str] = []
    for cand in pool:
        out.append(cand)
        if len(out) >= N:
            break
        # mutation 1: add "reference earlier turns if relevant"
        alt = f"{cand} Please reference earlier turns if relevant."
        if alt not in out:
            out.append(alt)
            if len(out) >= N:
                break
        # mutation 2: tiny synonyms
        alt2 = cand.replace("Summarize", "Provide a summary of").replace("Describe", "Detail")
        if alt2 not in out:
            out.append(alt2)
            if len(out) >= N:
                break

    # pad by SPO-lite if still short
    if len(out) < N:
        more = spo_hook(qid=qid, turn=turn, base=base, N=N - len(out), seeds=None)
        for m in more:
            if m not in out:
                out.append(m)
                if len(out) >= N:
                    break
    return out[:N]

# ----------------------------
# Mode mapping
# ----------------------------
def apply_mode_to_config(cfg_opt: Dict[str, Any], mode: str) -> None:
    """
    Mutate optimizer config in place based on the chosen mode.
    """
    cfg_opt.setdefault("entropy", {})
    cfg_opt.setdefault("dialogue_mode", "continuous")
    cfg_opt.setdefault("max_rounds", 3)
    cfg_opt.setdefault("sampling", {"candidates_per_turn": 3, "answers_per_question": 1})
    cfg_opt.setdefault("judge", {"use_context": True})
    cfg_opt.setdefault("opro_history_scope", "run")

    if mode == "none":
        cfg_opt["mode"] = "none"
        cfg_opt["strategy_family"] = "none"
        cfg_opt["max_rounds"] = 1
        cfg_opt["sampling"]["candidates_per_turn"] = 1
        cfg_opt["entropy"]["enabled"] = False
    elif mode == "spo":
        cfg_opt["mode"] = "spo"
        cfg_opt["strategy_family"] = "spo"
        cfg_opt["entropy"]["enabled"] = False
    elif mode == "entropy":
        cfg_opt["mode"] = "entropy"
        cfg_opt["strategy_family"] = "spo"
        cfg_opt["entropy"]["enabled"] = True
        cfg_opt["entropy"].setdefault("metric", "entropy")
    elif mode == "opro":
        cfg_opt["mode"] = "opro"
        cfg_opt["strategy_family"] = "opro"
        cfg_opt["entropy"]["enabled"] = False
    elif mode == "opro_entropy":
        cfg_opt["mode"] = "opro_entropy"
        cfg_opt["strategy_family"] = "opro"
        cfg_opt["entropy"]["enabled"] = True
        cfg_opt["entropy"].setdefault("metric", "entropy")
    elif mode == "current":
        cfg_opt["mode"] = "current"
        cfg_opt["strategy_family"] = "spo"
        cfg_opt["entropy"]["enabled"] = True
        cfg_opt["entropy"].setdefault("metric", "entropy")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def bind_hook_by_mode(opt: PromptOptimizer, mode: str) -> None:
    """
    Attach candidate generator hook to optimizer by mode.
    """
    if mode == "none":
        opt._gen_candidates_hook = lambda qid, turn, base, N, seeds=None: [base]
    elif mode in ("spo", "entropy", "current"):
        opt._gen_candidates_hook = lambda qid, turn, base, N, seeds=None: spo_hook(qid, turn, base, N, seeds=None)
    elif mode in ("opro", "opro_entropy"):
        # run-scope seeds are handled inside PromptOptimizer; we pass through seeds argument
        opt._gen_candidates_hook = lambda qid, turn, base, N, seeds=None: opro_hook(qid, turn, base, N, seeds=seeds)
    else:
        # no hook -> use optimizer's default minimal variants
        pass

# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run multi-turn prompt optimizer (continuous dialogue).")
    ap.add_argument("--mode", required=True,
                    choices=["none", "spo", "opro", "entropy", "opro_entropy", "current"],
                    help="Optimization strategy.")
    ap.add_argument("--template", default=None, help="Template YAML (contains qa + optimizer knobs).")
    ap.add_argument("--project", required=True, help="Project name; outputs to workspace/<project>.")
    ap.add_argument("--config", required=True, help="Models/endpoints YAML.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None, help="Optional log file path.")
    ap.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    ap.add_argument("--dataset", choices=["template", "mt-bench"], default="template",
                    help="Use qa from template or load MT-Bench dataset.")
    ap.add_argument("--mtbench-dir", default="data/mt-bench",
                    help="Local dir for MT-Bench cache/read.")
    ap.add_argument("--mtbench-split", default="train")
    ap.add_argument("--mtbench-limit", type=int, default=None)
    ap.add_argument("--mtbench-categories", default="",
                    help="Comma-separated categories, e.g. writing,reasoning")
    return ap

def main() -> int:
    args = build_argparser().parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(args.log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Load template (optimizer config + qa)
    tmpl: Dict[str, Any] = {}
    if args.template:
        tmpl = load_config_any(args.template)
    if not tmpl:
        tmpl = {}

    if args.dataset == "mt-bench":
        rows = load_or_download_mtbench(args.mtbench_dir, split=args.mtbench_split)
        categories = [x.strip() for x in args.mtbench_categories.split(",") if x.strip()]
        qa = mtbench_rows_to_qa(rows, limit=args.mtbench_limit, categories=categories)
    else:
        qa = tmpl.get("qa") or []
    if not qa:
        raise ValueError("No QA items loaded. Provide --template with qa or --dataset mt-bench.")

    # Make a DEEP copy of template as optimizer config
    cfg_opt = json.loads(json.dumps(tmpl))
    # Ensure essentials
    # Mode -> config switches
    apply_mode_to_config(cfg_opt, args.mode)

    # Load model endpoints and build clients
    mcfg = load_config_any(args.config)
    exec_llm = build_client(mcfg, "execute_model", "exec-llm")
    eval_llm = build_client(mcfg, "evaluate_model", "judge-llm")
    opt_llm = build_client(mcfg, "optimize_model", "opt-llm")

    # Workspace
    workdir = Path(args.output_dir) if args.output_dir else (Path("workspace") / args.project)
    ensure_dir(workdir)
    with open(workdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "dataset": args.dataset,
            "template": args.template,
            "mtbench_dir": args.mtbench_dir,
            "mtbench_split": args.mtbench_split,
            "mtbench_limit": args.mtbench_limit,
            "project": args.project,
        }, f, ensure_ascii=False, indent=2)

    # Init optimizer
    opt = PromptOptimizer(
        config=cfg_opt,
        exec_llm=exec_llm,
        eval_llm=eval_llm,
        opt_llm=opt_llm,
        workdir=str(workdir),
    )

    # Attach hook by mode
    bind_hook_by_mode(opt, args.mode)

    # Optimize
    best_dir = opt.optimize(qa=qa)
    print(f"[OK] Best round artifacts -> {best_dir}")
    print(f"[OK] Best per-turn rewrites -> {workdir}/best_prompts_by_turn.json")
    print(f"[OK] Run config -> {workdir}/run_config.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
