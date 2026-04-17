# Prompt Optimizer: Training-Free Prompt Evolution with Uncertainty Signals

This repository implements a **training-free prompt evolution** framework that improves prompts over multiple rounds using **LLM-as-a-judge** scoring and an **uncertainty proxy** (entropy/diversity) computed from multiple sampled responses. The system targets **multi-turn dialogue tasks** and supports both baseline prompt evolution (SPO-/OPRO-like) and entropy-aware variants used in the accompanying paper.

At a high level, each optimization round:

1. **Proposes** prompt candidates.
2. **Executes** each candidate prompt on a fixed set of multi-turn QA items (optionally sampling multiple answers).
3. **Evaluates** answers with a judge model to produce scalar quality scores.
4. **Computes** an uncertainty signal (entropy/diversity proxy) from sampled answers.
5. **Adjusts** candidate scores using a preference-aware shaping rule, then **selects** the best prompt per turn and per round.

---

## Repository Structure

* `components/`

  * `prompt_optimizer.py`
    Core optimizer: multi-round, turn-aware prompt evolution; executes prompts, evaluates candidates, applies entropy shaping, and writes artifacts.
  * `entropy_utils.py`
    **Semantic entropy proxy** via sentence-embedding clustering (with lexical fallback). Use this module for experiments that require clustering-based semantic entropy.
* `utils/`

  * `llm_client.py`
    OpenAI-compatible chat client (works with OpenAI-compatible servers such as vLLM).
* `scripts/`

  * `run_optimizer.py`
    Recommended entrypoint for optimization runs (loads YAML template, selects mode, runs optimizer).
  * `posthoc_compare_groups.py`
    Post-hoc aggregation across multiple runs in a workspace directory.
  * `render_curves.py`
    Utility to render score/CI curves from `curves.jsonl` (if present).
* `demo/`, `writing-demo/`
  Minimal runnable multi-turn templates (recommended starting points).
* `config/`
  Model endpoint configs (OpenAI-compatible) and other settings.

---

## Installation

### 1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Optional: enable semantic entropy clustering

If you want embedding-based semantic clustering entropy (instead of a lightweight proxy), install:

```bash
pip install sentence-transformers numpy torch
```

---

## Model Endpoints Configuration

Edit `config/models.yaml` to point to your **OpenAI-compatible** chat endpoints.

Example (OpenAI-compatible self-hosted endpoint):

```yaml
execute_model:
  base_url: "http://service.org.cn/v1"
  api_key: "EMPTY"
  model: "deepseek-v3.2"
  timeout_s: 120
  max_retries: 3

evaluate_model:
  base_url: "http://127.0.0.1:8001/v1"
  api_key: "EMPTY"
  model: "Qwen2.5-7B-Instruct"
  timeout_s: 120
  max_retries: 3

optimize_model:
  base_url: "http://127.0.0.1:8002/v1"
  api_key: "EMPTY"
  model: "Qwen2.5-7B-Instruct"
  timeout_s: 120
  max_retries: 3
```

Notes:

* `execute_model` generates task answers.
* `evaluate_model` acts as the judge (returns a scalar score).
* `optimize_model` proposes prompt candidates.

---

## Task Templates

### Multi-turn QA schema (recommended)

See `writing-demo/demo.yaml` for a minimal example. A typical QA item looks like:

```yaml
qa:
  - qid: "writing_001"
    turns:
      - user: "Write a short email to request a meeting."
  - qid: "writing_002"
    turns:
      - user: "Draft a two-paragraph project update for stakeholders."
```

Key fields:

* `prompt`: initial prompt instruction to evolve
* `requirements`: evaluation criteria shown to the judge
* `max_rounds`: number of evolution rounds
* `sampling`: sampling parameters used during execution (answers per question/turn, temperature/top_p, etc.)

---

## Running the Optimizer

### Quickstart

From the repo root:

```bash
python -m scripts.run_optimizer \
  --mode entropy \
  --template writing-demo/demo.yaml \
  --project writing_demo_entropy \
  --config config/models.yaml
```

### MT-Bench batch mode (auto download + local cache reuse)

```bash
python -m scripts.run_optimizer \
  --mode current \
  --dataset mt-bench \
  --mtbench-dir data/mt-bench \
  --mtbench-split train \
  --project mtbench_current \
  --config config/models.yaml \
  --output-dir workspace/mtbench_current \
  --log-file workspace/mtbench_current/run.log
```

If `--mtbench-dir` already contains cached `mt_bench_<split>.jsonl`, it is loaded directly (no re-download).


### One-command MT-Bench suite (Original / SPO / OPRO / Ours)

Use `scripts/run_mtbench_suite.py` to run four strategies with a single command:

```bash
python -m scripts.run_mtbench_suite \
  --template writing-demo/demo.yaml \
  --config config/models.yaml \
  --mtbench-dir data/mt-bench \
  --mtbench-split train \
  --output-root workspace/mtbench_suite
```

Mode aliases:

* `original` -> `none` (no optimization, one-pass baseline)
* `spo` -> SPO-like
* `opro` -> OPRO-like
* `ours` -> `current`

Per-mode logs/results are separated into timestamped subfolders under `--output-root`, and each run still writes `run_config.json`, `best_prompt_scores.json`, `best_prompts_by_turn.json`, plus detailed rollout logs.

### Modes

* `spo`: entropy shaping disabled; SPO-like lightweight candidate mutation.
* `opro`: entropy shaping disabled; OPRO-like optimization using prompt/score history.
* `entropy`: SPO-style evolution + entropy/diversity preference shaping.
* `opro_entropy`: OPRO-style evolution + entropy/diversity preference shaping.

Example:

```bash
python -m scripts.run_optimizer \
  --mode opro_entropy \
  --template writing-demo/demo.yaml \
  --project writing_demo_opro_entropy \
  --config config/models.yaml
```

---

## Score Adjustment: Entropy/Diversity Preference Shaping

For each candidate prompt, the system computes:

* `score_raw`: judge score for the sampled answers.
* `feature`: uncertainty proxy from sampled answers (entropy or diversity).

Then it applies preference-aware shaping:

* If the task/turn is **exploration-prefer**, the shaped score increases with the feature.
* If the task/turn is **stability-prefer**, the shaped score decreases with the feature.

A simplified form is:

```text
score_shaped = score_raw + alpha * sign(prefer) * feature
```

Preference can be inferred online (e.g., correlation between candidate scores and features within the same turn) or specified in the template.

---

## Semantic Entropy Proxy (Clustering-Based)

`components/entropy_utils.py` implements a semantic entropy proxy:

1. Embed sampled outputs with SentenceTransformers.
2. Build a similarity graph and derive clusters.
3. Compute **normalized cluster entropy** in `[0, 1]` (higher means more semantic dispersion).
4. Fall back to lexical entropy if embeddings are unavailable.

### Using semantic entropy in the optimizer

The optimizer can use either a lightweight entropy proxy or the clustering-based semantic entropy proxy. To use semantic entropy, compute the feature via:

```python
from components.entropy_utils import calculate_entropy
feature = calculate_entropy(sampled_outputs)
```

---

## Outputs and Artifacts

Each run writes to:

```
workspace/<project>/
  prompts/
    meta.json
    results.json
    curves.jsonl
    round_1/
      prompt.txt
      answers.txt
    round_2/
      ...
```

Typical artifacts:

* `results.json`: per-round candidate prompts, scores, features, and selections.
* `curves.jsonl`: per-round accepted score and optional bootstrap summaries.
* `round_k/prompt.txt`, `round_k/answers.txt`: accepted prompt and its executed answers.

---

## Post-hoc Analysis

### Compare multiple runs in a workspace

```bash
python -m scripts.posthoc_compare_groups \
  --root workspace \
  --out workspace/posthoc_summary.json
```

---

## Practical Tips

* Use `answers_per_question > 1` (or multi-sampling) to make entropy/diversity signals meaningful.
* Keep judge instructions strict and deterministic (e.g., numeric-only outputs).
* Tune `max_rounds` and `n_candidates` conservatively to control compute cost.
