# components/prompt_optimizer.py
# -*- coding: utf-8 -*-
import os, json, time, re, random, uuid, logging
from typing import Dict, Any, List, Optional, Tuple
import yaml

from utils.llm_client import LLMClient
from components.entropy_utils import calculate_entropy, summarize_diversity
from components.stats import bootstrap_mean_ci

log = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Unified optimizer supporting four modes:
    - "spo"           : single-step comparison based (old vs new)
    - "opro"          : OPRO-style trajectory optimization (prompt,score history)
    - "entropy"       : SPO + entropy tendency feedback
    - "opro_entropy"  : OPRO + entropy/diversity shaping
    """

    def __init__(self, config: Dict[str, Any], mode: str, template_file: str, project_name: str):
        mode = mode.lower().strip()
        assert mode in {"spo", "opro", "entropy", "opro_entropy"}, f"Unknown mode: {mode}"
        self.mode = mode
        self.cfg = config

        # LLM clients
        self.opt_llm = LLMClient(config["optimize_model"])
        self.eval_llm = LLMClient(config["evaluate_model"])
        self.exec_llm = LLMClient(config["execute_model"])

        # Template
        with open(template_file, "r", encoding="utf-8") as f:
            self.template = yaml.safe_load(f)
        self.current_prompt: str = self.template.get("prompt", "")
        self.requirements: str = self.template.get("requirements", "")
        self.qa: List[Dict[str, str]] = self.template.get("qa", [])
        self.max_rounds: int = int(self.template.get("max_rounds", 10))
        self.success_threshold: Optional[float] = self.template.get("success_threshold", None)
        self.opro_cfg = self.template.get("opro", {"n_candidates": 2, "keep_top_k": 5})
        self.spo_cfg = self.template.get("spo", {"accept_on_tie": False})
        self.entropy_cfg = self.template.get("entropy", {"high_entropy_threshold": 0.55, "low_entropy_threshold": 0.20})
        self.opro_entropy_cfg = self.template.get("opro_entropy", {
            "enabled": True,
            "alpha": 10.0,
            "metric": "entropy",   # or "diversity"
            "prefer": "auto",      # 'auto' | 'open' | 'closed' | 'none'
            "high_entropy_threshold": 0.55,
            "low_entropy_threshold": 0.20
        })

        # Workspace
        self.project = project_name
        self.workdir = os.path.join("workspace", project_name, "prompts")
        os.makedirs(self.workdir, exist_ok=True)

        # Curves file path (append one line per round)
        self.curves_path = os.path.join(self.workdir, "curves.jsonl")

        # State
        self.results_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []  # for OPRO-family
        self.last_answers: Optional[List[str]] = None
        self.round_i: int = 0

        # write minimal meta
        try:
            meta_path = os.path.join(self.workdir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "project": project_name,
                    "mode": self.mode,
                    "template": template_file,
                    "timestamp": time.time(),
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("Failed to write meta.json: %s", e)

    # ---------- Utilities ----------
    def _save_round_artifacts(self, rdir: str, prompt: str, answers: List[str]):
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(os.path.join(rdir, "answers.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(answers))

    def _append_log(self, entry: Dict[str, Any]):
        self.results_log.append(entry)
        with open(os.path.join(self.workdir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results_log, f, ensure_ascii=False, indent=2)

    # ---------- Diversity scalarization ----------
    def _diversity_scalar(self, diversity_obj: Any, answers: List[str]) -> float:
        """
        Turn a possibly-structured diversity summary into a single comparable scalar.
        - If it's already numeric, cast to float.
        - If it's a dict, prefer common keys: unique_ratio/distinct_ratio/ttr/score/value/shannon/simpson/ratio/unique_token_ratio
        - Fallback: unique/total ratio on answers.
        """
        if isinstance(diversity_obj, (int, float)):
            try:
                return float(diversity_obj)
            except Exception:
                pass
        if isinstance(diversity_obj, dict):
            for k in ("unique_ratio", "distinct_ratio", "ttr", "score", "value", "shannon", "simpson", "ratio", "unique_token_ratio"):
                v = diversity_obj.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
        try:
            u = len(set(a.strip() for a in answers if isinstance(a, str)))
            n = len(answers)
            s = (u / n) if n else 0.0
            return max(0.0, min(1.0, float(s)))
        except Exception:
            return 0.0

    # ---------- OPRO+Entropy shaping ----------
    def _opro_entropy_adjust(self, raw_score: float, entropy: float, diversity_scalar: float) -> float:
        """
        Compute adjusted score for OPRO+Entropy:
          score = raw_score + alpha * delta(signal)
        where signal is entropy or diversity_scalar.
        """
        cfg = self.opro_entropy_cfg or {}
        if not cfg or not cfg.get("enabled", True):
            return float(raw_score)
        prefer = str(cfg.get("prefer", "auto")).lower().strip()
        alpha = float(cfg.get("alpha", 10.0))
        hi = float(cfg.get("high_entropy_threshold", self.entropy_cfg.get("high_entropy_threshold", 0.55)))
        lo = float(cfg.get("low_entropy_threshold", self.entropy_cfg.get("low_entropy_threshold", 0.20)))
        metric = str(cfg.get("metric", "entropy")).lower().strip()
        # choose signal
        signal = float(entropy) if metric == "entropy" else float(diversity_scalar)
        # auto-detect open/closed from requirements
        if prefer == "auto":
            is_open = ("open-ended" in self.requirements.lower()) or ("creative" in self.requirements.lower())
            prefer = "open" if is_open else "closed"
        if prefer == "open":
            delta = signal - hi
        elif prefer == "closed":
            delta = (lo - signal)
        else:
            return float(raw_score)
        return float(raw_score) + alpha * float(delta)

    # ---------- Bootstrap-aware helpers ----------
    def _execute_prompt_grouped(self, prompt: str) -> List[List[str]]:
        """
        Execute prompt and return per-question answer lists: [[ans1..k], [ans1..k], ...]
        """
        grouped: List[List[str]] = []
        samp = self.template.get("sampling", {}) or {}
        k = int(samp.get("answers_per_question", 1))
        use_n = bool(samp.get("use_n", True))
        temp_override = samp.get("temperature", None)
        top_p_override = samp.get("top_p", None)
        for qa in self.qa:
            question = qa.get("question", "").strip()
            if not question:
                grouped.append([]); continue
            user = f"{prompt}\n\nQuestion:\n{question}\nAnswer:"
            messages = self.exec_llm.make_messages(system_prompt=None, user_prompt=user)
            q_answers: List[str] = []
            if k <= 1:
                out = self.exec_llm.chat_complete(
                    messages,
                    **({} if temp_override is None else {"temperature": temp_override}),
                    **({} if top_p_override is None else {"top_p": top_p_override}),
                )
                q_answers.append((out or "").strip())
            else:
                if use_n:
                    try:
                        outs = self.exec_llm.chat_complete(
                            messages, n=k, return_choices=True,
                            **({} if temp_override is None else {"temperature": temp_override}),
                            **({} if top_p_override is None else {"top_p": top_p_override}),
                        )
                        q_answers.extend([o.strip() for o in (outs or [])])
                    except Exception:
                        for _ in range(k):
                            out = self.exec_llm.chat_complete(
                                messages,
                                **({} if temp_override is None else {"temperature": temp_override}),
                                **({} if top_p_override is None else {"top_p": top_p_override}),
                            )
                            q_answers.append((out or "").strip())
                else:
                    for _ in range(k):
                        out = self.exec_llm.chat_complete(
                            messages,
                            **({} if temp_override is None else {"temperature": temp_override}),
                            **({} if top_p_override is None else {"top_p": top_p_override}),
                        )
                        q_answers.append((out or "").strip())
            grouped.append(q_answers)
        return grouped

    def _score_one_question(self, question: str, answers: List[str]) -> float:
        """
        Ask evaluator to rate one question's answers on 0..100.
        """
        instr = (
            "Rate the overall quality of the following answers from 0 to 100 (higher is better). "
            "Consider correctness, completeness, and alignment with requirements. "
            "Respond with ONLY a number on the first line.\n\n"
            f"Requirements:\n{self.requirements}\n\n"
            f"Question:\n{question}\n\n"
            "Answers:\n" + "\n".join(f"- {a}" for a in answers) + "\n\nScore (0-100):"
        )
        msg = self.eval_llm.make_messages(system_prompt="Be precise and output a single number.", user_prompt=instr)
        raw = (self.eval_llm.chat_complete(msg) or "").strip()
        m = re.search(r"(\d+(\.\d+)?)", raw)
        score = float(m.group(1)) if m else 0.0
        return max(0.0, min(100.0, float(score)))

    # ---------- Execution & Evaluation ----------
    def _execute_prompt(self, prompt: str) -> List[str]:
        """
        Legacy flat execution used in some paths; grouped is preferred.
        """
        answers: List[str] = []
        samp = self.template.get("sampling", {}) or {}
        k = int(samp.get("answers_per_question", 1))
        use_n = bool(samp.get("use_n", True))
        temp_override = samp.get("temperature", None)
        top_p_override = samp.get("top_p", None)

        for qa in self.qa:
            question = qa.get("question", "").strip()
            if not question:
                continue
            user = f"{prompt}\n\nQuestion:\n{question}\nAnswer:"
            messages = self.exec_llm.make_messages(system_prompt=None, user_prompt=user)

            if k <= 1:
                out = self.exec_llm.chat_complete(
                    messages,
                    **({} if temp_override is None else {"temperature": temp_override}),
                    **({} if top_p_override is None else {"top_p": top_p_override}),
                )
                answers.append((out or "").strip())
            else:
                if use_n:
                    try:
                        outs = self.exec_llm.chat_complete(
                            messages, n=k, return_choices=True,
                            **({} if temp_override is None else {"temperature": temp_override}),
                            **({} if top_p_override is None else {"top_p": top_p_override}),
                        )
                        answers.extend([o.strip() for o in (outs or [])])
                    except Exception:
                        for _ in range(k):
                            out = self.exec_llm.chat_complete(
                                messages,
                                **({} if temp_override is None else {"temperature": temp_override}),
                                **({} if top_p_override is None else {"top_p": top_p_override}),
                            )
                            answers.append((out or "").strip())
                else:
                    for _ in range(k):
                        out = self.exec_llm.chat_complete(
                            messages,
                            **({} if temp_override is None else {"temperature": temp_override}),
                            **({} if top_p_override is None else {"top_p": top_p_override}),
                        )
                        answers.append((out or "").strip())
        return answers

    def _judge_pairwise(self, old_answers: List[str], new_answers: List[str]) -> Tuple[str, float]:
        """
        Ask evaluator to pick A (old) or B (new). Return tuple (winner, confidence_score_0_1).
        """
        compare_instr = (
            "You are a strict impartial judge. Compare two sets of answers A (old) and B (new) "
            "against the task requirements below. Decide which set is better overall.\n"
            "Return ONLY a single letter: 'A' or 'B'. If truly indistinguishable, return 'T' for tie.\n\n"
            f"Requirements:\n{self.requirements}\n\n"
            "Set A (old):\n" + "\n".join(f"- {a}" for a in old_answers) + "\n\n"
            "Set B (new):\n" + "\n".join(f"- {b}" for b in new_answers) + "\n\n"
            "Your verdict (A/B/T):"
        )
        msg = self.eval_llm.make_messages(system_prompt="Be concise and deterministic.", user_prompt=compare_instr)
        verdict = (self.eval_llm.chat_complete(msg) or "").strip().upper()
        winner = "B" if verdict.startswith("B") else ("A" if verdict.startswith("A") else "T")
        conf = 1.0 if winner in {"A", "B"} else 0.5
        return winner, conf

    def _score_answers(self, answers: List[str]) -> float:
        """
        Ask evaluator to rate the set of answers 0..100 (float).
        """
        instr = (
            "Rate the overall quality of the following answers from 0 to 100 (higher is better). "
            "Consider correctness, completeness, and alignment with requirements. "
            "Respond with ONLY a number on the first line.\n\n"
            f"Requirements:\n{self.requirements}\n\n"
            "Answers:\n" + "\n".join(f"- {a}" for a in answers) + "\n\nScore (0-100):"
        )
        msg = self.eval_llm.make_messages(system_prompt="Be precise and output a single number.", user_prompt=instr)
        raw = (self.eval_llm.chat_complete(msg) or "").strip()
        m = re.search(r"(\d+(\.\d+)?)", raw)
        score = float(m.group(1)) if m else 0.0
        return max(0.0, min(100.0, float(score)))

    # ---------- OPRO ask builder ----------
    def _build_opro_ask(self, recent_hist: List[Dict[str, Any]], nonce: str, slot_idx: int) -> str:
        """
        Build the OPRO ask with history and a diversity nonce.
        """
        hist_txt = "\n".join(
            f'Prompt:\n"""\n{h.get("prompt","")}\n"""\nScore: {h.get("score",0)}\n---'
            for h in recent_hist
        )
        ask = (
            "We have a history of prompts and their scores. "
            "Propose a new, improved prompt instruction that maximizes the score.\n"
            "Return ONLY a single-line JSON object of the exact form {\"prompt\":\"...\"} "
            "or {\"candidates\":[\"...\",\"...\"]}.\n"
            "Rules:\n"
            "- Do NOT include code fences, triple quotes, 'Prompt:', 'Score:', comments, or any metadata.\n"
            "- Do NOT copy or mention the diversity token below.\n"
            "- Keep the prompt concise, actionable, and self-contained.\n\n"
            f"{hist_txt}\n\n"
            f"Diversity token (do not copy): {nonce}\n\n"
            f"(candidate slot: {slot_idx})\n"
            "New prompt JSON:"
        )
        return ask

    # ---------- Optimizer Prompts ----------
    def _opro_propose(self, slot_idx: int = 0) -> str:
        """
        Propose a new prompt (OPRO-style), with:
          - optimize_sampling: temperature/top_p/n/seed/stop/max_tokens/history_k/retry/response_format
          - per-call fresh nonce (cache-busting) + candidate slot index
          - structured JSON output (prefer response_format=json_object) + robust fallback
          - same-round & recent-history dedup; retries with new nonce; safe fallback
        """
        osamp = (self.template.get("optimize_sampling") or {})
        temp = osamp.get("temperature", 0.7)
        top_p = osamp.get("top_p", 0.9)
        use_n = bool(osamp.get("use_n", True))
        n = int(osamp.get("n", 1) or 1)
        seed_cfg = osamp.get("seed", None)
        stop = osamp.get("stop", None)
        max_tokens = osamp.get("max_tokens", None)
        hist_k = int(osamp.get("history_k", max(1, len(self.history))))
        max_retry = int(osamp.get("retry", 2))
        resp_fmt = osamp.get("response_format", None)  # e.g., {"type":"json_object"}

        recent_hist = self.history[-hist_k:] if hist_k > 0 else self.history

        def _normalize(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def _extract_prompts(text: str, nonce: str) -> List[str]:
            if not text:
                return []
            t = text.strip()
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.S | re.I)
            if m:
                t = m.group(1).strip()
            # JSON first
            try:
                obj = json.loads(t)
                outs: List[str] = []
                if isinstance(obj, dict):
                    if "prompt" in obj and isinstance(obj["prompt"], str):
                        outs = [obj["prompt"]]
                    elif "candidates" in obj and isinstance(obj["candidates"], list):
                        outs = [x for x in obj["candidates"] if isinstance(x, str)]
                outs = [(x or "").replace(nonce, "").strip() for x in outs if (x or "").strip()]
                return outs
            except Exception:
                pass
            # non-JSON cleanup
            t = re.sub(r"^```.*?$", "", t, flags=re.M)
            t = t.replace('"""', "").replace("'''", "")
            t = re.sub(r"^\s*Prompt:.*?$", "", t, flags=re.M | re.I)
            t = re.sub(r"^\s*Score:\s*\d+(\.\d+)?\s*$", "", t, flags=re.M | re.I)
            t = re.sub(r"^\s*---\s*$", "", t, flags=re.M)
            t = re.sub(re.escape(nonce), "", t, flags=re.I)
            t = t.strip()
            t = re.sub(r'^[\'"]|[\'"]$', "", t).strip()
            return [t] if t else []

        # same-round & history dedup sets
        recent_prompts = {_normalize(h.get("prompt", "")) for h in recent_hist}
        seen_round = self._round_seen_norm

        attempt = 0
        while attempt <= max_retry:
            attempt += 1
            nonce = uuid.uuid4().hex
            ask = self._build_opro_ask(recent_hist, nonce, slot_idx)
            msg = self.opt_llm.make_messages(
                system_prompt="You are an expert prompt engineer. Output valid minified JSON only.",
                user_prompt=ask
            )

            kwargs = {}
            if temp is not None: kwargs["temperature"] = temp
            if top_p is not None: kwargs["top_p"] = top_p
            if max_tokens is not None: kwargs["max_tokens"] = max_tokens
            if stop: kwargs["stop"] = stop
            if resp_fmt: kwargs["response_format"] = resp_fmt  # JSON mode if backend supports

            # jitter seed if configured (avoid same RNG stream)
            if seed_cfg is not None:
                try:
                    jitter = (self.round_i << 12) ^ int(time.time() * 1e3) ^ random.randint(0, 2**16 - 1) ^ slot_idx
                    kwargs["seed"] = int(seed_cfg) ^ jitter
                except Exception:
                    pass

            # prefer multi-return in one call if supported
            outs_text: List[str] = []
            if use_n and n > 1 and hasattr(self.opt_llm, "chat_complete_n"):
                raw_list = self.opt_llm.chat_complete_n(msg, n=n, **kwargs)  # most backends won't have this
                outs_text = []
                for raw in (raw_list or []):
                    outs_text.extend(_extract_prompts(raw or "", nonce))
            else:
                raw = self.opt_llm.chat_complete(msg, **kwargs) or ""
                outs_text = _extract_prompts(raw, nonce)

            # normalize + dedup (vs. same-round + recent history)
            uniq: List[str] = []
            local_seen = set()
            for s in outs_text:
                k = _normalize(s)
                if not k or k in local_seen or k in seen_round or k in recent_prompts:
                    continue
                local_seen.add(k)
                uniq.append(s)

            # record into same-round seen
            for k in local_seen:
                seen_round.add(k)

            if uniq:
                return uniq[0]

            # retry with a new loop (fresh nonce & time-based jitter)
            log.debug("[OPRO] propose retry=%d (empty/dup), slot=%d", attempt, slot_idx)

        # ----- Second-chance ask (non-JSON fallback before synthesizing) -----
        try:
            nonce2 = uuid.uuid4().hex
            hist_txt = "\n".join(
                f'Prompt:\n"""\n{h.get("prompt","")}\n"""\nScore: {h.get("score",0)}\n---'
                for h in recent_hist
            )
            ask2 = (
                "We have a history of prompts and their scores. "
                "Propose a new, improved prompt instruction that maximizes the score.\n"
                "Return ONLY the revised prompt text on a single line (no JSON, no quotes, no fences).\n"
                "- Do NOT include 'Prompt:' or 'Score:' or the token below.\n\n"
                f"{hist_txt}\n\n"
                f"Diversity token (do not copy): {nonce2}\n\n"
                "Revised prompt:"
            )
            msg2 = self.opt_llm.make_messages(
                system_prompt="You are an expert prompt engineer. Output just the prompt text.",
                user_prompt=ask2
            )
            kwargs2 = {}
            if temp is not None: kwargs2["temperature"] = temp
            if top_p is not None: kwargs2["top_p"] = top_p
            if max_tokens is not None: kwargs2["max_tokens"] = max_tokens
            if stop: kwargs2["stop"] = stop
            # jitter seed again
            if seed_cfg is not None:
                try:
                    jitter2 = (self.round_i << 10) ^ int(time.time() * 1e3) ^ random.randint(0, 2**16 - 1) ^ (slot_idx << 4)
                    kwargs2["seed"] = int(seed_cfg) ^ jitter2
                except Exception:
                    pass
            raw2 = self.opt_llm.chat_complete(msg2, **kwargs2) or ""
            s2 = raw2.replace(nonce2, "").strip()
            s2 = re.sub(r"^```.*?```$", "", s2, flags=re.S | re.M)
            s2 = s2.replace('"""', "").replace("'''", "")
            s2 = re.sub(r"^\s*Prompt:.*?$", "", s2, flags=re.M | re.I)
            s2 = re.sub(r"^\s*Score:\s*\d+(\.\d+)?\s*$", "", s2, flags=re.M | re.I)
            s2 = re.sub(r"^\s*---\s*$", "", s2, flags=re.M)
            s2 = re.sub(r'^[\'"]|[\'"]$', "", s2).strip()
            k2 = re.sub(r"\s+", " ", s2)
            if k2 and k2 not in seen_round and k2 not in recent_prompts:
                seen_round.add(k2)
                return s2
        except Exception:
            pass

        # fallback: still nothing → synthesize a safe variation
        base = self.current_prompt.strip() or "You are a helpful assistant."
        fallback = (
            base.rstrip() +
            "\n\n# Additions: use a 3-step structure (acknowledge → inquire → actionable next step); "
            "be concise and concrete; ensure a single-sentence final focus."
        )
        log.warning("[OPRO] fallback candidate synthesized for slot=%d", slot_idx)
        return fallback

    def _spo_propose(self, last_feedback: Optional[str]) -> str:
        ask = (
            "Improve the current prompt to better satisfy the requirements. "
            "Return ONLY the revised prompt text.\n\n"
            f"Current prompt:\n\"\"\"\n{self.current_prompt}\n\"\"\"\n\n"
            f"Requirements:\n{self.requirements}\n"
        )
        if last_feedback:
            ask += f"\nPrevious round note:\n{last_feedback}\n"
        ask += "\nNew prompt:"
        msg = self.opt_llm.make_messages(system_prompt="You are an expert prompt engineer.", user_prompt=ask)
        return (self.opt_llm.chat_complete(msg) or "").strip()

    # ---------- Main optimize loop ----------
    def optimize(self):
        """
        Side effects:
          - Writes workspace/<project>/prompts/round_{i}/(prompt.txt, answers.txt)
          - Appends per-round JSON entries to results.json
          - Appends a line to curves.jsonl per round
        Returns:
          - Path to workspace/<project>/prompts/results.json
        """
        # If OPRO-family, evaluate initial prompt and seed history
        last_feedback: Optional[str] = None
        if self.mode in {"opro", "opro_entropy"}:
            init_answers = self._execute_prompt(self.current_prompt)
            init_score_raw = float(self._score_answers(init_answers))
            if self.mode == "opro_entropy":
                init_ent = float(calculate_entropy(init_answers))
                init_div_raw = summarize_diversity(init_answers)
                init_div_s = float(self._diversity_scalar(init_div_raw, init_answers))
                init_score = float(self._opro_entropy_adjust(init_score_raw, init_ent, init_div_s))
            else:
                init_score = init_score_raw
            self.history.append({"prompt": self.current_prompt, "score": float(init_score)})

        for r in range(1, self.max_rounds + 1):
            self.round_i = r
            round_dir = os.path.join(self.workdir, f"round_{r}")
            os.makedirs(round_dir, exist_ok=True)

            # same-round dedup pool for OPRO proposals
            self._round_seen_norm = set()

            # 1) Propose new prompt(s)
            if self.mode in {"opro", "opro_entropy"}:
                candidates = []
                n_cand = int(self.opro_cfg.get("n_candidates", 2))
                for i_slot in range(max(1, n_cand)):
                    cand = self._opro_propose(slot_idx=i_slot)
                    if cand and cand.strip():
                        candidates.append(cand.strip())
            else:
                new_prompt = self._spo_propose(last_feedback)
                candidates = [new_prompt.strip()] if new_prompt and new_prompt.strip() else []

            if not candidates:
                # Defensive: avoid empty candidate set
                round_entry = {
                    "round": r,
                    "mode": self.mode,
                    "status": "no_candidates",
                    "error": "No prompt candidates generated by optimizer model."
                }
                self._append_log(round_entry)
                break

            # 2) Execute + evaluate candidates
            candidate_logs: List[Dict[str, Any]] = []
            for cand in candidates:
                # Execute grouped for per-QA scoring & flatten for compatibility
                grouped = self._execute_prompt_grouped(cand)
                ans = [a for g in grouped for a in g]

                # ---- answers_by_qid: ALWAYS BUILD EARLY (avoid UnboundLocalError) ----
                answers_by_qid: List[Dict[str, Any]] = []
                for i_q, (qa_item, g) in enumerate(zip(self.qa, grouped)):
                    qid = qa_item.get("id") or qa_item.get("qid") or f"q{i_q+1}"
                    answers_by_qid.append({
                        "qid": qid,
                        "question": qa_item.get("question", ""),
                        "answers": g
                    })

                # diversity signals
                ent = float(calculate_entropy(ans))
                div_raw = summarize_diversity(ans)
                div_s = float(self._diversity_scalar(div_raw, ans))

                if self.mode in {"opro", "opro_entropy"}:
                    # raw OPRO score for selection baseline
                    score_raw = float(self._score_answers(ans))
                    score = score_raw
                    if self.mode == "opro_entropy":
                        score = float(self._opro_entropy_adjust(score_raw, ent, div_s))

                    # per-question bootstrap CI (OPRO family only)
                    per_q_scores: List[float] = []
                    for qa_item, group_ans in zip(self.qa, grouped):
                        q = qa_item.get("question", "")
                        if not q.strip():
                            continue
                        per_q_scores.append(float(self._score_one_question(q, group_ans if group_ans else [""])))
                    if per_q_scores:
                        mean_s, lo, hi = bootstrap_mean_ci(per_q_scores, confidence=0.95, iterations=1000, seed=1234)
                        ci = {"mean": float(mean_s), "ci95": [float(lo), float(hi)], "n": len(per_q_scores)}
                    else:
                        ci = {"mean": float(score_raw), "ci95": [float(score_raw), float(score_raw)], "n": 1}

                    candidate_logs.append({
                        "prompt": cand,
                        "answers": ans,
                        "answers_by_qid": answers_by_qid,
                        "entropy": float(ent),
                        "diversity": div_raw,              # keep full dict
                        "diversity_scalar": float(div_s),  # scalar for shaping/analysis
                        "score": float(score),             # used for selection
                        "score_raw": float(score_raw),     # for analysis
                        "per_question_scores": per_q_scores,
                        "bootstrap": ci,
                    })
                else:
                    # SPO / Entropy: keep pairwise judging behavior
                    if self.last_answers is not None:
                        winner, _conf = self._judge_pairwise(self.last_answers, ans)
                        score = 1.0 if winner == "B" else (0.5 if winner == "T" else 0.0)
                        score_raw = float(self._score_answers(ans))
                    else:
                        score_raw = float(self._score_answers(ans))
                        score = 1.0 if score_raw >= 50.0 else 0.0
                    # 为便于后续分析，这里同样给出 per-q 打分与 bootstrap
                    per_q_scores: List[float] = []
                    for qa_item, group_ans in zip(self.qa, grouped):
                        q = qa_item.get("question", "")
                        if not q.strip():
                            continue
                        per_q_scores.append(float(self._score_one_question(q, group_ans if group_ans else [""])))
                    if per_q_scores:
                        mean_s, lo, hi = bootstrap_mean_ci(per_q_scores, confidence=0.95, iterations=1000, seed=1234)
                        ci = {"mean": float(mean_s), "ci95": [float(lo), float(hi)], "n": len(per_q_scores)}
                    else:
                        ci = {"mean": float(score_raw), "ci95": [float(score_raw), float(score_raw)], "n": 1}

                    candidate_logs.append({
                        "prompt": cand,
                        "answers": ans,
                        "answers_by_qid": answers_by_qid,
                        "entropy": float(ent),
                        "diversity": div_raw,
                        "diversity_scalar": float(div_s),
                        "score": float(score),
                        "score_raw": float(score_raw),
                        "per_question_scores": per_q_scores,
                        "bootstrap": ci,
                    })

            # 2.5) Select best candidate (before any use of best_*)
            best_entry = max(candidate_logs, key=lambda c: c.get("score", float("-inf")))
            best_prompt: str = best_entry["prompt"]
            best_answers: List[str] = best_entry["answers"]
            best_score: float = float(best_entry["score"])
            best_boot: Optional[Dict[str, Any]] = best_entry.get("bootstrap")

            # 3) Entropy feedback (entropy mode only: SPO + entropy tendency)
            adjusted_record = None
            if self.mode == "entropy":
                ent_best = float(best_entry.get("entropy", 0.0))
                hi = float(self.entropy_cfg.get("high_entropy_threshold", 0.55))
                lo = float(self.entropy_cfg.get("low_entropy_threshold", 0.20))
                is_open = ("open-ended" in self.requirements.lower()) or ("creative" in self.requirements.lower())

                adjusted_prompt = None
                if (not is_open and ent_best > hi):
                    last_feedback = "Answers exhibited high variance; make the prompt more specific and constrain output format."
                    adjusted_prompt = best_prompt + (
                        "\n\n# Guidance: Be precise and deterministic. "
                        "Use step-by-step reasoning but provide a single final answer."
                    )
                elif (is_open and ent_best < lo):
                    last_feedback = "Answers were too similar; encourage diversity and originality."
                    adjusted_prompt = best_prompt + (
                        "\n\n# Guidance: Encourage multiple perspectives and creative phrasing while staying relevant."
                    )
                else:
                    last_feedback = None

                # If changed, re-execute to keep consistency; reuse score or rescore if desired
                if adjusted_prompt is not None and adjusted_prompt.strip() != best_prompt:
                    best_prompt = adjusted_prompt.strip()
                    best_answers = self._execute_prompt(best_prompt)
                    # 保持同一轮的判别逻辑一致性，不强制重评
                    best_boot = best_boot  # keep

            # 4) Accept/update state
            if self.mode in {"opro", "opro_entropy"}:
                status = "added"
            else:
                status = "improved" if (best_score > 0.5 or self.last_answers is None) else "rejected"
                if status == "rejected" and self.spo_cfg.get("accept_on_tie", False) and best_score == 0.5:
                    status = "improved"

            if status in {"improved", "added"}:
                self.current_prompt = best_prompt
                self.last_answers = best_answers

            # 5) OPRO: update history & keep top-K
            if self.mode in {"opro", "opro_entropy"}:
                self.history.append({"prompt": best_prompt, "score": float(best_score)})
                keep_k = int(self.opro_cfg.get("keep_top_k", 5))
                self.history = sorted(self.history, key=lambda x: x["score"], reverse=True)[:keep_k]

            # 6) Persist round artifacts & log
            self._save_round_artifacts(round_dir, best_prompt, best_answers)
            round_entry: Dict[str, Any] = {
                "round": r,
                "mode": self.mode,
                "status": status,
                "accepted_prompt": best_prompt,
                "accepted_score": float(best_score),
                "accepted_score_raw": (float(best_entry.get("score_raw")) if "score_raw" in best_entry else None),
                "accepted_bootstrap": best_boot,
                "answers": best_answers,
                "candidates": candidate_logs,
            }
            if self.mode in {"opro", "opro_entropy"}:
                round_entry["history"] = self.history
            self._append_log(round_entry)

            # curves line
            try:
                with open(self.curves_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "round": r,
                        "accepted_score": float(best_score),
                        "bootstrap_mean": (float(best_boot["mean"]) if isinstance(best_boot, dict) and "mean" in best_boot else None),
                        "bootstrap_ci95": (best_boot.get("ci95") if isinstance(best_boot, dict) else None)
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                log.warning("Failed to append curves.jsonl: %s", e)

            # 7) Early stopping
            if self.mode in {"opro", "opro_entropy"} and self.success_threshold is not None and best_score >= float(self.success_threshold):
                break
            if self.mode in {"spo", "entropy"} and status == "rejected":
                break

            log.info("Round %d finished; candidates=%d; status=%s", r, len(candidates), status)

        # Done
        return os.path.join(self.workdir, "results.json")
