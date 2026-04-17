# -*- coding: utf-8 -*-
"""
Continuous-Dialogue Prompt Optimizer (pairwise-enhanced for SPO/entropy)

- SPO / entropy modes:
  * Multi-candidate generation per round
  * Pairwise A/B/T judging with reasons & alignment points
  * Winner-advances tournament
  * Reasons/alignment fed to opt_llm to produce next-round rewrites
  * Entropy/diversity shaping used as tie-breaker or mild re-ranking

- OPRO / OPRO-entropy modes: keep existing hook-based generator.

This file is a drop-in replacement for your current prompt_optimizer.py.
"""
from __future__ import annotations

import re
import json
import math
import shutil
import string
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger("prompt_optimizer")

# -------------------------------
# Numerics
# -------------------------------
def safe_pearsonr(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xm = sum(xs[:n]) / n
    ym = sum(ys[:n]) / n
    num = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n))
    dx = math.sqrt(sum((xs[i] - xm) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - ym) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))

def text_entropy_proxy(text: str) -> float:
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    total = 0
    for ch in text:
        if ch in string.whitespace:
            continue
        counts[ch] = counts.get(ch, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent

def answers_diversity_scalar(answers: List[str]) -> float:
    if not answers:
        return 0.0
    sets = []
    for a in answers:
        toks = [w.strip(string.punctuation).lower() for w in a.split() if w.strip(string.punctuation)]
        sets.append(set(toks))
    if len(sets) == 1:
        return 0.0
    dsum, cnt = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            d = 1.0 - (len(inter) / (len(u) + 1e-12))
            dsum += d; cnt += 1
    return dsum / cnt if cnt else 0.0

# -------------------------------
# Data
# -------------------------------
@dataclass
class TurnRecord:
    rewrite: str
    answers: List[str]
    score_raw: float
    entropy: float
    diversity_scalar: float
    score_shaped: float

@dataclass
class RunState:
    store: Dict[str, Dict[str, Dict[int, TurnRecord]]] = field(default_factory=dict)
    def set_turn(self, qid: str, t: int, rec: TurnRecord) -> None:
        self.store.setdefault(qid, {}).setdefault("turns", {})[t] = rec
    def get_turn(self, qid: str, t: int) -> Optional[TurnRecord]:
        return self.store.get(qid, {}).get("turns", {}).get(t)

# -------------------------------
# Optimizer
# -------------------------------
class PromptOptimizer:
    def __init__(self, config: Dict[str, Any], exec_llm: Any, eval_llm: Any,
                 workdir: str = "./workdir", opt_llm: Any = None) -> None:
        self.cfg = config
        self.exec_llm = exec_llm
        self.eval_llm = eval_llm
        self.opt_llm = opt_llm or exec_llm
        self.workdir = pathlib.Path(workdir); self.workdir.mkdir(parents=True, exist_ok=True)

        self.dialogue_mode = self.cfg.get("dialogue_mode", "continuous")
        self.max_rounds = int(self.cfg.get("max_rounds", 3))
        samp = self.cfg.get("sampling", {})
        self.candidates_per_turn = int(samp.get("candidates_per_turn", 3))
        self.answers_per_question = int(samp.get("answers_per_question", 1))

        self.entropy_cfg = self.cfg.get("entropy", {
            "enabled": True, "alpha": 8.0, "metric": "entropy", "corr_threshold": 0.10, "history_k": 20,
        })
        self.judge_cfg = self.cfg.get("judge", {"use_context": True})
        self.opro_scope = self.cfg.get("opro_history_scope", "run")

        # traces for corr over rounds
        self.turn_trace: Dict[int, Tuple[List[float], List[float]]] = {}
        self.global_turn_memory: Dict[int, List[Tuple[str, float]]] = {}
        self.system_prompt = self.cfg.get("system_prompt", "You are a helpful assistant.")
        self.eval_system_prompt = "Output a short JSON with keys: winner (A|B|T), reasons, alignment_points (array)."
        self.strategy_family = str(self.cfg.get("strategy_family", "spo")).lower().strip()

    # ---------- public ----------
    def optimize(self, qa: List[Dict[str, Any]]) -> pathlib.Path:
        if not qa:
            raise ValueError("Empty QA set.")
        T_max = max(len(x.get("turns", [])) for x in qa)
        if T_max <= 0:
            raise ValueError("No turns found.")

        best_run_state: Optional[RunState] = None
        best_total_score = float("-inf")
        rounds_meta: List[Dict[str, Any]] = []

        logger.info("=== optimization start | rounds=%d | candidates/turn=%d | answers/k=%d ===",
                    self.max_rounds, self.candidates_per_turn, self.answers_per_question)

        for r in range(self.max_rounds):
            run_state = RunState(); total_score = 0.0
            round_dir = self.workdir / f"rollout_{r}"
            if round_dir.exists(): shutil.rmtree(round_dir)
            round_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[rollout %d] BEGIN", r)

            for t in range(T_max):
                turn_dir = round_dir / f"turn_{t}"; turn_dir.mkdir(exist_ok=True)
                logger.info("[rollout %d | turn %d] BEGIN", r, t)

                for item in qa:
                    qid = item["qid"]; turns = item["turns"]
                    if t >= len(turns): continue
                    orig_q = turns[t]
                    
                    if self._is_spo_family():
                        guidance = {"reasons": [], "points": []}
                        candidates = self._spo_candidates_multi(orig_q, guidance, N=self.candidates_per_turn, round_id=r, qid=qid, turn=t)
                    else:
                        candidates = self._gen_candidates(qid, t, orig_q, run_state, N=self.candidates_per_turn)

                    cand_logs: List[Dict[str, Any]] = []
                    for ci, cand in enumerate(candidates):
                        answers = self._exec_answers_with_context(qid, t, cand, run_state, k=self.answers_per_question)
                        raw = self._score_one_turn(qid, t, cand, answers, run_state)
                        ent = self._feature_entropy_or_diversity(answers, prefer="entropy")
                        div = self._feature_entropy_or_diversity(answers, prefer="diversity")
                        cand_logs.append({
                            "idx": ci, "rewrite": cand, "answers": answers,
                            "score_raw": float(raw), "entropy": float(ent), "diversity_scalar": float(div)
                        })
                        logger.info("[rollout %d | turn %d | %s] cand#%d raw=%.2f ent=%.3f div=%.3f rw='%s'",
                                    r, t, qid, ci, raw, ent, div, self._short(cand))

                    prefer = self._decide_turn_preference(t, cand_logs)
                    alpha = float(self.entropy_cfg.get("alpha", 8.0)) if self.entropy_cfg.get("enabled", True) else 0.0
                    metric = self.entropy_cfg.get("metric", "entropy")
                    for c in cand_logs:
                        feat = c["entropy"] if metric == "entropy" else c["diversity_scalar"]
                        sgn = 1.0 if prefer == "open" else (-1.0 if prefer == "closed" else 0.0)
                        c["score_shaped"] = c["score_raw"] + alpha * sgn * feat

                    if self._is_spo_family():
                        winner_log, aggr = self._pairwise_tournament(qid, t, cand_logs, prefer, r)
                        self._remember_guidance(qid, t, aggr)
                        best = winner_log
                    else:
                        best = max(cand_logs, key=lambda x: x["score_shaped"])

                    total_score += best.get("score_shaped", best.get("score_raw", 0.0))

                    rec = TurnRecord(
                        rewrite=best["rewrite"], answers=best["answers"],
                        score_raw=float(best["score_raw"]),
                        entropy=float(best["entropy"]),
                        diversity_scalar=float(best["diversity_scalar"]),
                        score_shaped=float(best.get("score_shaped", best["score_raw"]))
                    )
                    run_state.set_turn(qid, t, rec)

                    ent_like = rec.entropy if metric == "entropy" else rec.diversity_scalar
                    self._push_turn_trace(t, ent_like, rec.score_raw)

                    self._dump_json(turn_dir / f"{qid}.cand_logs.json", cand_logs)
                    self._dump_json(turn_dir / f"{qid}.best.json", {
                        "rewrite": rec.rewrite, "answers": rec.answers,
                        "score_raw": rec.score_raw, "score_shaped": rec.score_shaped,
                        "preference": prefer, "metric": metric,
                    })

                    logger.info("[rollout %d | turn %d | %s] WIN raw=%.2f shaped=%.2f ent=%.3f div=%.3f rw='%s'",
                                r, t, qid, rec.score_raw, rec.score_shaped, rec.entropy, rec.diversity_scalar, self._short(rec.rewrite))

                logger.info("[rollout %d | turn %d] END", r, t)

            rounds_meta.append({"round": r, "total_score": total_score})
            self._dump_json(round_dir / "_round_summary.json", rounds_meta[-1])
            logger.info("[rollout %d] END | total_score=%.2f", r, total_score)

            if total_score > best_total_score:
                best_total_score = total_score; best_run_state = run_state
                best_dir = self.workdir / "best_run"
                if best_dir.exists(): shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                self._persist_best_run(best_dir, qa, best_run_state)

        self._dump_json(self.workdir / "_meta.json", {"best_total_score": best_total_score, "rounds": rounds_meta})
        logger.info("=== optimization end | best_total_score=%.2f ===", best_total_score)
        return self.workdir / "best_run"

    # ---------- execution/scoring ----------
    def _build_context_messages(self, qid: str, t: int, candidate_rewrite: str, run_state: RunState) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.dialogue_mode == "continuous":
            prev = run_state.store.get(qid, {}).get("turns", {})
            for i in range(t):
                if i in prev:
                    pr = prev[i]
                    msgs.append({"role": "user", "content": pr.rewrite})
                    ctx_ans = pr.answers[0] if pr.answers else ""
                    msgs.append({"role": "assistant", "content": ctx_ans})
        msgs.append({"role": "user", "content": candidate_rewrite})
        return msgs

    def _exec_messages(self, messages: List[Dict[str, str]], k: int = 1) -> List[str]:
        outs: List[str] = []
        for _ in range(max(1, k)):
            try:
                txt = self.exec_llm.chat_complete(messages=messages)
            except Exception:
                txt = ""
            outs.append(txt if isinstance(txt, str) else str(txt))
        return outs

    def _exec_answers_with_context(self, qid: str, t: int, candidate_rewrite: str, run_state: RunState, k: int = 1) -> List[str]:
        messages = self._build_context_messages(qid, t, candidate_rewrite, run_state)
        return self._exec_messages(messages, k=k)

    def _score_one_turn(self, qid: str, t: int, candidate_rewrite: str, answers: List[str], run_state: RunState) -> float:
        if not self.judge_cfg.get("use_context", True):
            prompt = self._mk_eval_prompt_noctx(candidate_rewrite, answers)
        else:
            prompt = self._mk_eval_prompt_withctx(qid, t, candidate_rewrite, answers, run_state)
        messages = [{"role": "system", "content": "Be precise. Output a NUMBER 0-100 only."},
                    {"role": "user", "content": prompt}]
        try:
            raw = self.eval_llm.chat_complete(messages=messages)
        except Exception:
            raw = "0"
        m = re.search(r"-?\d+(?:\.\d+)?", str(raw) or "")
        num = float(m.group(0)) if m else 0.0
        return max(0.0, min(100.0, num))

    def _mk_eval_prompt_withctx(self, qid: str, t: int, cand: str, answers: List[str], run_state: RunState) -> str:
        prev_ctx = []
        prev = run_state.store.get(qid, {}).get("turns", {})
        for i in range(t):
            if i in prev:
                pr = prev[i]
                prev_ctx.append(f"Turn {i} — User: {pr.rewrite}\nTurn {i} — Assistant: {pr.answers[0] if pr.answers else ''}")
        ctx_blob = "\n\n".join(prev_ctx) if prev_ctx else "(no previous turns)"
        ans_blob = "\n".join(f"- {a}" for a in answers)
        return ("Evaluate the assistant answers (0-100). Higher is better.\n"
                f"Conversation context:\n{ctx_blob}\n\n"
                f"Current user (rewrite):\n{cand}\n\n"
                f"Candidate answers:\n{ans_blob}\n\n"
                "Only output a NUMBER (0-100).")

    def _mk_eval_prompt_noctx(self, cand: str, answers: List[str]) -> str:
        ans_blob = "\n".join(f"- {a}" for a in answers)
        return ("Evaluate the assistant answers (0-100). Higher is better.\n"
                f"Question (rewrite):\n{cand}\n\n"
                f"Candidate answers:\n{ans_blob}\n\nOnly output a NUMBER (0-100).")

    # ---------- preference & shaping ----------
    def _feature_entropy_or_diversity(self, answers: List[str], prefer: str = "entropy") -> float:
        if prefer == "diversity":
            return answers_diversity_scalar(answers)
        if not answers:
            return 0.0
        return sum(text_entropy_proxy(a) for a in answers) / len(answers)

    def _decide_turn_preference(self, turn_idx: int, cand_logs: List[Dict[str, Any]]) -> str:
        th = float(self.entropy_cfg.get("corr_threshold", 0.10))
        metric = self.entropy_cfg.get("metric", "entropy")
        xs_now = [(c["entropy"] if metric == "entropy" else c["diversity_scalar"]) for c in cand_logs]
        ys_now = [c["score_raw"] for c in cand_logs]
        corr_now = safe_pearsonr(xs_now, ys_now) if len(cand_logs) >= 3 else 0.0

        hist_ent, hist_scores = self.turn_trace.get(turn_idx, ([], []))
        H = int(self.entropy_cfg.get("history_k", 20))
        corr_hist = safe_pearsonr(hist_ent[-H:], hist_scores[-H:]) if len(hist_ent) >= 6 else 0.0

        if abs(corr_now) >= th:   return "open" if corr_now > 0 else "closed"
        if abs(corr_hist) >= th:  return "open" if corr_hist > 0 else "closed"
        return "none"

    def _push_turn_trace(self, turn_idx: int, ent_like: float, raw_score: float) -> None:
        a, b = self.turn_trace.get(turn_idx, ([], []))
        a.append(float(ent_like)); b.append(float(raw_score))
        self.turn_trace[turn_idx] = (a, b)

    # ---------- SPO/entropy: multi-candidate + pairwise + feedback ----------
    def _is_spo_family(self) -> bool:
        return self.strategy_family == "spo"

    def _spo_candidates_multi(self, base: str, guidance: Dict[str, Any], N: int, round_id: int, qid: str, turn: int) -> List[str]:
        base = (base or "").strip()
        if not base: return [base]
        reasons = "; ".join(guidance.get("reasons", [])[-4:])
        points  = "; ".join(guidance.get("points", [])[-6:])
        sys = """ROLE: Prompt Rewriter (instruction-level).
        You rewrite user INSTRUCTIONS into alternative INSTRUCTIONS for a model.

        CRITICAL:
        - Output MUST be alternative instructions (prompts to a model), NOT answers or content that fulfills the task.
        - Preserve meaning; keep each instruction self-contained; do not add new requirements.

        FORMAT:
        - Return ONLY a JSON array of strings (no keys, no code fences)

        DIVERSITY:
        - Produce distinct phrasings (avoid trivial synonyms-only duplicates).

        STYLE AXES (for diversity WITHOUT changing requirements):
        - CLOSE-style: determinate, narrowly scoped, imperative, minimal hedging.
        - OPEN-style: exploratory tone, invites perspective while keeping the SAME three focus areas, no extra tasks.
        - NEUTRAL: plain, professional, balanced.
        Ensure ≥⌈0.3·N⌉ CLOSE-style and ≥⌈0.3·N⌉ OPEN-style; the rest NEUTRAL. Do NOT label styles in text.
        Produce distinct phrasings (not trivial synonym swaps).
        """

        user = (
            f"INPUT_INSTRUCTION:\n{base}\n\n"
            f"CONTEXT (optional):\n"
            f"- Reasons from previous comparisons: {reasons or '(none)'}\n"
            f"- Alignment points to emphasize: {points or '(none)'}\n\n"
            f"TASK:\n"
            f"Rewrite the INPUT_INSTRUCTION into EXACTLY {N} distinct alternative INSTRUCTIONS (not answers), "
            f"each suitable to give directly to a model as a directive.\n\n"
            f"CONSTRAINTS:\n"
            f"- Preserve all original requirements\n"
            f"OUTPUT:\n"
            f"Return ONLY the JSON array of strings."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        outs: List[str] = []
        try:
            txt = self.opt_llm.chat_complete(messages=msgs, temperature=0.7)
            arr = self._safe_parse_json_list(txt)
            if not arr:
                raise ValueError("empty-json")
            outs = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
        except Exception:
            outs = self._spo_lite_variants(base, N)
        uniq = []
        sset = set()
        for x in outs:
            if x not in sset:
                uniq.append(x); sset.add(x)
        if not uniq: uniq = [base]
        if len(uniq) < N:
            uniq.extend(self._spo_lite_variants(base, N - len(uniq)))
        uniq = uniq[:N]
        logger.info("[r%d|%s|t%d] generated %d SPO candidates (opt_llm=%s)",
                    round_id, qid, turn, len(uniq), "yes" if self.opt_llm else "no")
        for i, x in enumerate(uniq):
            logger.info("  - cand#%d: %s", i, self._short(x))
        return uniq

    def _spo_lite_variants(self, base: str, need: int) -> List[str]:
        base = base.strip()
        pool = [
            base,
            f"{base} Please answer directly and reference earlier turns if relevant.",
            base.replace("Describe", "Detail").replace("Explain", "Clarify"),
            base + " Keep your wording concrete and specific." if len(base.split()) > 8 else base,
        ]
        out = []
        for x in pool:
            if x not in out:
                out.append(x)
            if len(out) >= need: break
        while len(out) < need:
            out.append(base + f" [v{len(out)}]")
        return out[:need]

    def _safe_parse_json_list(self, txt: str) -> List[str]:
        try:
            js = json.loads(txt)
            if isinstance(js, list): return js
        except Exception:
            pass

        m = re.search(r"\[.*\]", txt, flags=re.S)
        if not m: return []
        try:
            js = json.loads(m.group(0));  return js if isinstance(js, list) else []
        except Exception:
            return []
        
    def _zstats(self, vals):
        vals = [float(v) for v in vals]
        n = len(vals)
        if n == 0:
            return [], 0.0, 1.0
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / max(1, n - 1)
        sigma = (var ** 0.5) or 1.0
        z = [(v - mu) / sigma for v in vals]
        return z, mu, sigma

    def _rank_candidates_pre_match(self, cand_logs: list, prefer: str, metric: str, cfg: dict):
        """Return new list sorted by a strong rank_score that emphasizes shaped."""
        if not cand_logs:
            return cand_logs

        beta_shaped = float(cfg.get("beta_shaped", 1.0))
        beta_feat   = float(cfg.get("beta_feature", 0.35))
        beta_raw    = float(cfg.get("beta_raw", 0.15))
        jitter_eps  = float(cfg.get("jitter_eps", 1e-3))
        prefer_none_fallback = float(cfg.get("prefer_none_feature_boost", 0.5))
        if str(prefer).lower() == "none":
            beta_feat *= (1.0 + prefer_none_fallback)

        shaped = [c.get("score_shaped", c.get("score_raw", 0.0)) for c in cand_logs]
        raw    = [c.get("score_raw", 0.0) for c in cand_logs]
        feat   = [(c["entropy"] if metric == "entropy" else c["diversity_scalar"]) for c in cand_logs]

        z_shaped, _, _ = self._zstats(shaped)
        z_raw,    _, _ = self._zstats(raw)
        z_feat,   _, _ = self._zstats(feat)

        ranked = []
        for i, c in enumerate(cand_logs):
            rs = (beta_shaped * z_shaped[i]) + (beta_feat * z_feat[i]) + (beta_raw * z_raw[i])
            rs += jitter_eps * (hash(c["rewrite"]) % 997) / 997.0
            ranked.append((rs, c))
            logger.debug("[rank] #%d rs=%.4f z_sh=%.3f z_f=%.3f z_raw=%.3f | rw='%s'",
                        i, rs, z_shaped[i], z_feat[i], z_raw[i], self._short(c["rewrite"]))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _pairwise_tournament(self, qid: str, t: int, cand_logs: list, prefer: str, r: int):
        metric = self.entropy_cfg.get("metric", "entropy")
        rank_cfg = (self.cfg.get("ranking") or {})
        pool = self._rank_candidates_pre_match(cand_logs, prefer=prefer, metric=metric, cfg=rank_cfg)

        strat = str(rank_cfg.get("pairing_strategy", "snake")).lower().strip()
        def make_pairs(arr):
            n = len(arr)
            if strat == "snake" and n > 2:
                pairs = []
                for i in range(n // 2):
                    pairs.append((arr[i], arr[n - 1 - i]))
                if n % 2 == 1:
                    pairs.append((arr[n // 2], None))
                return pairs
            else:
                pairs = []
                i = 0
                while i < n:
                    a = arr[i]
                    b = arr[i + 1] if i + 1 < n else None
                    pairs.append((a, b))
                    i += 2
                return pairs

        agg = {"reasons": [], "points": []}
        entropy_on = bool(self.entropy_cfg.get("enabled", True))
        HL, RESET = "\033[1;33m", "\033[0m"

        stage = 0
        while len(pool) > 1:
            stage += 1
            pairs = make_pairs(pool)
            logger.info("[pairwise] stage=%d | pairs=%d | strategy=%s", stage, len([p for p in pairs if p[1] is not None]), strat)

            next_pool = []
            for A, B in pairs:
                if B is None:
                    next_pool.append(A)
                    continue

                verdict = self._judge_pairwise(qid, t, A, B)
                win = verdict.get("winner", "T")
                reasons = verdict.get("reasons", "")
                points  = verdict.get("alignment_points", []) or []

                if win == "A":
                    chosen = A
                elif win == "B":
                    chosen = B
                else:
                    a_shaped = float(A.get("score_shaped", A.get("score_raw", 0.0)))
                    b_shaped = float(B.get("score_shaped", B.get("score_raw", 0.0)))
                    chosen = A if a_shaped >= b_shaped else B
                    win = "A" if chosen is A else "B"
                    msg_plain = (f"[TIE→SHAPED] prefer={prefer} metric={metric} "
                                f"pick={win} shaped(A)={a_shaped:.2f} shaped(B)={b_shaped:.2f}")
                    try:
                        logger.info("%s%s%s", HL, msg_plain, RESET) if entropy_on else logger.info(msg_plain)
                    except Exception:
                        logger.info(msg_plain)

                logger.info("[A/B/T] %s vs %s -> %s | reason='%s' | points=%s",
                            self._short(A['rewrite']), self._short(B['rewrite']), win,
                            self._short(reasons, 160), "; ".join(points[:4]))
                agg["reasons"].append(reasons)
                agg["points"].extend(points[:4])
                next_pool.append(chosen)
            pool = next_pool

        winner = pool[0]
        agg["reasons"] = [self._short(x, 300) for x in agg["reasons"][-6:]]
        agg["points"]  = list(dict.fromkeys([self._short(p, 80) for p in agg["points"]]))[:8]
        return winner, agg

    def _judge_pairwise(self, qid: str, t: int, A: Dict[str, Any], B: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of A/B/T judgement on eval_llm and returns JSON
        {winner: "A|B|T", reasons: "...", alignment_points: ["..."]}
        """
        sys = self.eval_system_prompt
        user = (
            "You are a careful evaluator for pairwise prompt selection.\n\n"
            f"Prompt A:\n{A['rewrite']}\n\nAnswers A:\n- " + "\n- ".join(A["answers"]) + "\n\n"
            f"Prompt B:\n{B['rewrite']}\n\nAnswers B:\n- " + "\n- ".join(B["answers"]) + "\n\n"
            "Choose the better prompt for the task given the answers' usefulness and faithfulness.\n"
            "Return a compact JSON with keys: winner (A|B|T), reasons (string), alignment_points (array of short strings)."
        )
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        try:
            txt = self.eval_llm.chat_complete(messages=messages, temperature=0.2, max_tokens=256)
        except Exception:
            txt = ""
        try:
            js = json.loads(txt)
            if not isinstance(js, dict): raise ValueError
            winner = js.get("winner", "T")
            reasons = js.get("reasons", "")
            points = js.get("alignment_points", []) or []
            if winner not in ("A","B","T"): winner = "T"
            if not isinstance(points, list): points = []
            return {"winner": winner, "reasons": str(reasons), "alignment_points": [str(p) for p in points]}
        except Exception:
            winner = "A" if A["score_raw"] > B["score_raw"] else ("B" if B["score_raw"] > A["score_raw"] else "T")
            return {"winner": winner, "reasons": "", "alignment_points": []}

    def _remember_guidance(self, qid: str, t: int, aggr: Dict[str, Any]) -> None:
        memo = " | ".join(aggr.get("points", [])[-6:])
        self.global_turn_memory.setdefault(t, []).append((memo, 0.0))

    # ---------- default generators & hooks ----------
    def _gen_candidates(self, qid: str, t: int, base_text: str, run_state: RunState, N: int = 3) -> List[str]:
        if hasattr(self, "_gen_candidates_hook") and callable(self._gen_candidates_hook):
            try:
                seeds_run = self._collect_run_turn_memory(t, run_state) if self.opro_scope == "run" else self.global_turn_memory.get(t, [])
                return self._gen_candidates_hook(qid=qid, turn=t, base=base_text, N=N, seeds=seeds_run)
            except Exception:
                pass
        cands = {base_text.strip()}
        if len(cands) < N: cands.add(self._neutral_paraphrase(base_text, 0))
        if len(cands) < N: cands.add(self._neutral_paraphrase(base_text, 1))
        if len(cands) < N: cands.add(self._neutral_paraphrase(base_text, 2))
        self.global_turn_memory.setdefault(t, []).append((base_text, 0.0))
        return list(cands)[:N]

    def _neutral_paraphrase(self, text: str, style: int = 0) -> str:
        text = text.strip()
        if style == 0: return text
        if style == 1: return f"{text} (keep answers concise and grounded in prior turns only)"
        return f"{text} Please answer directly and reference earlier turns if relevant."

    def _collect_run_turn_memory(self, t: int, run_state: RunState) -> List[Tuple[str, float]]:
        seeds: List[Tuple[str, float]] = []
        for qid, bucket in run_state.store.items():
            tr = bucket.get("turns", {}).get(t)
            if tr: seeds.append((tr.rewrite, tr.score_raw))
        return seeds

    # ---------- persistence ----------
    def _persist_best_run(self, best_dir: pathlib.Path, qa: List[Dict[str, Any]], run_state: RunState) -> None:
        convo_dump: Dict[str, Any] = {}
        best_prompts: Dict[str, Dict[int, str]] = {}
        flat_rows: List[Dict[str, Any]] = []
        for item in qa:
            qid = item["qid"]; turns = item["turns"]; convo_dump[qid] = []
            for t in range(len(turns)):
                rec = run_state.get_turn(qid, t)
                if rec is None: continue
                convo_dump[qid].append({
                    "turn": t, "rewrite": rec.rewrite, "answers": rec.answers,
                    "score_raw": rec.score_raw, "entropy": rec.entropy,
                    "diversity_scalar": rec.diversity_scalar, "score_shaped": rec.score_shaped,
                })
                best_prompts.setdefault(qid, {})[t] = rec.rewrite
                flat_rows.append({
                    "qid": qid,
                    "turn": t,
                    "prompt": rec.rewrite,
                    "score_raw": rec.score_raw,
                    "score_shaped": rec.score_shaped,
                    "entropy": rec.entropy,
                    "diversity_scalar": rec.diversity_scalar,
                })
        self._dump_json(best_dir / "dialogue_trace.json", convo_dump)
        self._dump_json(self.workdir / "best_prompts_by_turn.json", best_prompts)
        self._dump_json(self.workdir / "best_prompt_scores.json", flat_rows)

    def _dump_json(self, path: pathlib.Path, obj: Any) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("write json failed: %s -> %s", path, e)

    # ---------- small utils ----------
    def _short(self, s: str, n: int = 120) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s

# -------------------------------
# Stub client (kept for completeness)
# -------------------------------
class SimpleLLM:
    def __init__(self, name: str = "exec"): self.name = name
    def chat_complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return f"[stub-{self.name}] {m.get('content','')[:512]}"
        return f"[stub-{self.name}]"
