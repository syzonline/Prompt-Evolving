# scripts/posthoc_compare_groups.py
import argparse, json, os, time, itertools
import yaml
from utils.llm_client import LLMClient

def load_final_prompt(results_json):
    data = json.load(open(results_json, "r", encoding="utf-8"))
    if not data:
        raise ValueError(f"No rounds in {results_json}")
    return data[-1]["accepted_prompt"]

def exec_answers(exec_client, prompt, qa_list, temperature=None, top_p=None):
    outs = []
    for qa in qa_list:
        q = qa["question"]
        user = f"{prompt}\n\nQuestion:\n{q}\nAnswer:"
        msgs = exec_client.make_messages(None, user)
        kwargs = {}
        if temperature is not None: kwargs["temperature"] = temperature
        if top_p is not None: kwargs["top_p"] = top_p
        out = exec_client.chat_complete(msgs, **kwargs)
        outs.append(out.strip())
    return outs

def judge_pair(eval_client, requirements, A, B, judge_temp=0.0):
    # A vs B
    instr1 = (
      "You are an impartial judge. Compare two answer sets A and B "
      "to the requirements. Return ONLY 'A', 'B', or 'T' (tie).\n\n"
      f"Requirements:\n{requirements}\n\n"
      "Set A:\n" + "\n".join(f"- {x}" for x in A) + "\n\n"
      "Set B:\n" + "\n".join(f"- {x}" for x in B) + "\n\n"
      "Your verdict (A/B/T):"
    )
    v1 = eval_client.chat_complete(
        eval_client.make_messages("Be concise and deterministic.", instr1),
        temperature=judge_temp
    ).strip().upper()
    # B vs A (flip to reduce position bias; MT-Bench notes position bias and mitigation)  # noqa
    instr2 = (
      "You are an impartial judge. Compare two answer sets A and B "
      "to the requirements. Return ONLY 'A', 'B', or 'T' (tie).\n\n"
      f"Requirements:\n{requirements}\n\n"
      "Set A:\n" + "\n".join(f"- {x}" for x in B) + "\n\n"
      "Set B:\n" + "\n".join(f"- {x}" for x in A) + "\n\n"
      "Your verdict (A/B/T):"
    )
    v2 = eval_client.chat_complete(
        eval_client.make_messages("Be concise and deterministic.", instr2),
        temperature=judge_temp
    ).strip().upper()
    # votes for original A (A wins when verdict=='A' in first, and 'B' in flipped)
    a_votes = (1 if v1.startswith("A") else 0) + (1 if v2.startswith("B") else 0)
    b_votes = (1 if v1.startswith("B") else 0) + (1 if v2.startswith("A") else 0)
    ties    = (1 if v1.startswith("T") else 0) + (1 if v2.startswith("T") else 0)
    return a_votes, b_votes, ties

def judge_score(eval_client, requirements, answers, judge_temp=0.0):
    instr = (
      "Rate the following answers on a 0-100 scale (higher is better). "
      "Respond with ONLY a number.\n\n"
      f"Requirements:\n{requirements}\n\n"
      "Answers:\n" + "\n".join(f"- {x}" for x in answers) + "\n\nScore:"
    )
    raw = eval_client.chat_complete(
        eval_client.make_messages("Be precise and output a single number.", instr),
        temperature=judge_temp
    )
    import re
    m = re.search(r"(\d+(\.\d+)?)", raw)
    s = float(m.group(1)) if m else 0.0
    return max(0.0, min(100.0, s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--eval_template", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                   help="workspace/<RUN>/prompts/results.json ...")
    ap.add_argument("--metric", choices=["pairwise","score"], default="pairwise")
    ap.add_argument("--judge_temp", type=float, default=0.0)
    ap.add_argument("--exec_temp", type=float, default=None)
    ap.add_argument("--exec_top_p", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    eval_client = LLMClient(cfg["evaluate_model"])
    exec_client = LLMClient(cfg["execute_model"])

    tmpl = yaml.safe_load(open(args.eval_template, "r", encoding="utf-8"))
    global_reqs = tmpl.get("requirements", "")
    groups = tmpl["groups"]

    # Load final prompts for each run
    run_prompts = {}
    for r in args.runs:
        run_name = r.split("/workspace/")[-1].split("/prompts/")[0]
        run_prompts[run_name] = load_final_prompt(r)

    # Answer cache: answers[group][run] = list[str]
    answers = {g["name"]: {} for g in groups}

    # Generate answers per group
    for g in groups:
        gname = g["name"]
        reqs = g.get("requirements", global_reqs)
        qa = g["qa"]
        for run_name, prompt in run_prompts.items():
            ans = exec_answers(exec_client, prompt, qa, temperature=args.exec_temp, top_p=args.exec_top_p)
            answers[gname][run_name] = ans

    # Evaluate per group
    results = {"metric": args.metric, "groups": [], "overall": {}}
    per_run_weighted = {rn: 0.0 for rn in run_prompts}
    total_weight = 0.0

    for g in groups:
        gname = g["name"]
        weight = float(g.get("weight", 1.0))
        reqs = g.get("requirements", global_reqs)

        # Per-run score storage
        per_run = {rn: 0.0 for rn in run_prompts}
        if args.metric == "pairwise":
            # Tournament: average pairwise win rate vs others
            for a, b in itertools.combinations(run_prompts.keys(), 2):
                A = answers[gname][a]; B = answers[gname][b]
                av, bv, tv = judge_pair(eval_client, reqs, A, B, judge_temp=args.judge_temp)
                # Each pair contributes two votes (plus possible ties)
                total_votes = av + bv + tv
                # Avoid division by zero; count only decisive votes
                decisive = max(1, av + bv)
                a_rate = av / decisive
                b_rate = bv / decisive
                per_run[a] += a_rate
                per_run[b] += b_rate
            # Normalize by number of opponents
            opp = max(1, len(run_prompts) - 1)
            for rn in per_run: per_run[rn] /= opp

        else:  # numeric score 0-100 per run (direct scoring)
            for rn in run_prompts:
                s = judge_score(eval_client, reqs, answers[gname][rn], judge_temp=args.judge_temp)
                per_run[rn] = s

        # Aggregate
        results["groups"].append({"name": gname, "weight": weight, "per_run": per_run})
        # Weighted addition for overall
        for rn in per_run:
            per_run_weighted[rn] += weight * per_run[rn]
        total_weight += weight

    # Overall (macro or weighted macro)
    if total_weight <= 0: total_weight = 1.0
    overall = {rn: per_run_weighted[rn] / total_weight for rn in per_run_weighted}
    results["overall"] = overall

    # Persist
    outdir = os.path.join("workspace", "eval_groups")
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join(outdir, f"results_{ts}.json")
    cpath = os.path.join(outdir, f"summary_{ts}.csv")
    with open(jpath, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
    # CSV
    runs = list(run_prompts.keys())
    import csv
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["group","weight"] + runs
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for g in results["groups"]:
            row = {"group": g["name"], "weight": g["weight"]}; row.update(g["per_run"]); w.writerow(row)
        row = {"group": "OVERALL", "weight": total_weight}; row.update(overall); w.writerow(row)
    print(f"[OK] wrote\n  {jpath}\n  {cpath}")

if __name__ == "__main__":
    main()
