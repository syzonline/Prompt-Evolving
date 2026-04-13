# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_or_download_mtbench(data_dir: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load MT-Bench rows from local cache first; otherwise download from Hugging Face.
    """
    root = Path(data_dir)
    cache_file = root / f"mt_bench_{split}.jsonl"
    if cache_file.exists():
        return _read_jsonl(cache_file)

    root.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install `datasets` to download MT-Bench: pip install datasets") from e

    ds = load_dataset("philschmid/mt-bench", split=split)
    rows = [dict(x) for x in ds]
    _write_jsonl(cache_file, rows)
    return rows


def mtbench_rows_to_qa(
    rows: List[Dict[str, Any]],
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    cats = set([c.strip() for c in (categories or []) if c and c.strip()])
    qa: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if cats and str(row.get("category", "")).strip() not in cats:
            continue
        turns = row.get("turns") or []
        if isinstance(turns, str):
            turns = [turns]
        turns = [str(t).strip() for t in turns if str(t).strip()]
        if not turns:
            continue
        qid = row.get("question_id") or row.get("id") or f"mtbench_{i}"
        qa.append({"qid": str(qid), "turns": turns})
        if limit is not None and len(qa) >= int(limit):
            break
    return qa
