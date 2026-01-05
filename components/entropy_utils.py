# entropy_utils.py
from typing import List, Tuple
from collections import Counter
import math
import re

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    _ST_OK = False

import numpy as np

_EMB_MODEL = None
def _get_emb_model(name: str = "/path/to/all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None and _ST_OK:
        try:
            _EMB_MODEL = SentenceTransformer(name, device="cpu")
            
        except Exception:
            _EMB_MODEL = None
    return _EMB_MODEL

_TOKEN_RGX = re.compile(r"[A-Za-z0-9_]+")
def _simple_tokens(s: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RGX.finditer(s or "")]

def _jaccard_sim(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    u = len(A | B)
    if u == 0:
        return 0.0
    return len(A & B) / u

def _lexical_entropy(outputs: List[str]) -> float:
    tokens: List[str] = []
    for out in outputs:
        if not out:
            continue
        tokens.extend(out.split())
    n = len(tokens)
    if n == 0:
        return 0.0
    cnt = Counter(tokens)
    ent = 0.0
    for c in cnt.values():
        p = c / n
        ent -= p * math.log2(p)
    V = max(len(cnt), 1)
    max_ent = math.log2(V) if V > 1 else 1.0
    h = ent / max_ent
    return float(max(0.0, min(1.0, h)))

def _semantic_clusters(outputs: List[str]) -> List[int]:
    n = len(outputs)
    if n == 0:
        return []
    if n == 1:
        return [0]

    model = _get_emb_model()
    if model is None:
        return list(range(n))

    embs = model.encode(outputs, normalize_embeddings=True, convert_to_numpy=True)
    S_sem = (embs @ embs.T).astype(np.float32)
    np.fill_diagonal(S_sem, 1.0)

    toks = [_simple_tokens(x) for x in outputs]
    S_lex = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S_lex[i, i] = 1.0
        for j in range(i + 1, n):
            sim = _jaccard_sim(toks[i], toks[j])
            S_lex[i, j] = S_lex[j, i] = sim

    beta = 0.85
    S = beta * S_sem + (1.0 - beta) * S_lex
    S = np.clip(S, 0.0, 1.0)


    k = max(2, int(0.2 * (n - 1)))
    margin = 0.04
    A = np.zeros((n, n), dtype=bool)

    order = np.argsort(-S, axis=1)
    snn = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        nbrs_i = order[i, 1:k+1]
        thr_i = max(0.0, float(S[i, nbrs_i].mean()) - margin) if k > 0 else 0.8
        for j in range(i + 1, n):
            nbrs_j = order[j, 1:k+1]
            thr_j = max(0.0, float(S[j, nbrs_j].mean()) - margin) if k > 0 else 0.8

            shared = len(set(nbrs_i).intersection(set(nbrs_j)))
            snn[i, j] = snn[j, i] = shared

            cond = (S[i, j] >= thr_i) and (S[i, j] >= thr_j) and (shared >= max(1, k // 4))
            A[i, j] = A[j, i] = cond

    density = A.sum() / max(1, n * (n - 1))
    if density < 0.02:
        A |= (S >= (S.mean() + 0.5 * S.std()))
    elif density > 0.25:
        A &= (S >= (S.mean() + 1.0 * S.std()))

    visited = [False] * n
    labels = [-1] * n
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        labels[i] = cid
        while queue:
            u = queue.pop()
            for v in range(n):
                if not visited[v] and A[u, v]:
                    visited[v] = True
                    labels[v] = cid
                    queue.append(v)
        cid += 1

    if cid == 1 and n >= 3:
        min_pair = np.unravel_index(np.argmin(S + np.eye(n)), S.shape)
        a, b = int(min_pair[0]), int(min_pair[1])
        group_a, group_b = set([a]), set([b])
        for t in range(n):
            if t in (a, b):
                continue
            group_a.add(t) if S[t, a] >= S[t, b] else group_b.add(t)
        if 1 <= len(group_a) < n and 1 <= len(group_b) < n:
            labels = [0 if i in group_a else 1 for i in range(n)]
            cid = 2

    return labels

def calculate_entropy(outputs: List[str]) -> float:
    outs = [o for o in (outputs or []) if isinstance(o, str) and o.strip()]
    if len(outs) == 0:
        return 0.0
    if len(outs) == 1:
        return 0.0

    labels = _semantic_clusters(outs)
    if not labels or len(labels) != len(outs):
        return _lexical_entropy(outs)

    K = max(labels) + 1 if labels else 1
    if K <= 1:
        return 0.0
    cnt = Counter(labels)
    n = float(len(outs))
    probs = [c / n for c in cnt.values()]

    H = 0.0
    for p in probs:
        if p > 0:
            H -= p * math.log2(p)

    H_norm = H / math.log2(K)  # [0,1]
    try:
        model = _get_emb_model()
        embs = model.encode(outs, normalize_embeddings=True, convert_to_numpy=True)
        S = (embs @ embs.T).astype(np.float32)
        intra = []
        for k in range(K):
            idx = [i for i, z in enumerate(labels) if z == k]
            if len(idx) >= 2:
                Sk = S[np.ix_(idx, idx)]
                m = (Sk.sum() - np.trace(Sk)) / (len(idx) * (len(idx) - 1))
                intra.append(float(m))
        if intra:
            dispersion = max(0.0, min(1.0, 1.0 - float(np.mean(intra))))
            H_norm = float(min(1.0, max(0.0, H_norm + 0.15 * dispersion)))
    except Exception:
        pass

    return float(max(0.0, min(1.0, H_norm)))
