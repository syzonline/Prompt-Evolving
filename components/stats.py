
from __future__ import annotations
from typing import List, Dict, Tuple, Any, Iterable, Optional
import math, random, statistics

def bootstrap_mean_ci(samples: List[float], confidence: float = 0.95, iterations: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    if not samples:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(samples)
    boots = []
    for _ in range(iterations):
        resample = [samples[rng.randrange(0, n)] for __ in range(n)]
        boots.append(statistics.fmean(resample))
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = boots[int(alpha * iterations)]
    hi = boots[int((1.0 - alpha) * iterations) - 1]
    return (statistics.fmean(samples), lo, hi)

def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def compute_elo(matches, k: float = 20.0, init: float = 1500.0, iters: int = 8):
    players = set()
    for a, b, _ in matches:
        players.add(a); players.add(b)
    elo = {p: init for p in players}
    ms = list(matches)
    for _ in range(max(1, iters)):
        for a, b, s in ms:
            ea = _elo_expected(elo[a], elo[b]); eb = 1.0 - ea
            sa = s; sb = 1.0 - s if s in (0.0, 1.0) else 0.5
            elo[a] = elo[a] + k * (sa - ea)
            elo[b] = elo[b] + k * (sb - eb)
    return elo

def derive_pairwise_from_scores(score_table: Dict[str, Dict[str, float]], tie_eps: float = 1e-9):
    matches = []
    for task, row in score_table.items():
        players = list(row.keys())
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a, b = players[i], players[j]
                sa, sb = row.get(a), row.get(b)
                if sa is None or sb is None: 
                    continue
                if abs(sa - sb) <= tie_eps:
                    matches.append((a, b, 0.5))
                elif sa > sb:
                    matches.append((a, b, 1.0))
                else:
                    matches.append((b, a, 1.0))
    return matches
