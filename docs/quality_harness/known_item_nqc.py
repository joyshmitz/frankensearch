#!/usr/bin/env python
"""Known-item NQC dense down-weight A/B (mirrors the Rust real_hybrid_knownitem bench in
Python, runnable here). Queries = first ~10 words of held-out scifact docs; target = that doc.
Strongly committed-lexical (the query is a literal doc prefix) -> high NQC, so the thesis
predicts down-weighting the dense tier helps/neutral. pool-min-max fusion, semantic_weight
scaled per query by clip(1 - beta*CDF(nqc_cv(lex)), w_min, 1). Reports recall@10 + MRR vs beta.
"""
import json, os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
from model2vec import StaticModel
from rank_bm25 import BM25Okapi
import snowballstemmer

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be but by for if in into is it no not of on or such "
           "that the their then there these they this to was will with".split())
_stem = snowballstemmer.stemmer("english")
def tok(s): return _stem.stemWords([t for t in TOKEN.findall(s.lower()) if t not in STOP])

def load_corpus():
    docs = []
    for line in open(os.path.join(HERE, "scifact", "corpus.jsonl")):
        o = json.loads(line)
        docs.append((o.get("title", "") + " " + o.get("text", "")).strip())
    return docs

def nqc_cv(scores):
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0: return 0.0
    m = s.mean()
    if m <= 1e-10: return 0.0
    return float(max(s.var(), 0.0) ** 0.5 / m)

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)

def main():
    POOL, K, W_MIN, N_Q = 100, 10, 0.1, 400
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    docs = load_corpus()
    doc_emb = model.encode(docs, show_progress_bar=False)
    doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])

    # Queries = first 10 words of the first N_Q docs with >=10 words; target = that doc index.
    q_src, q_text = [], []
    for i, d in enumerate(docs):
        words = d.split()
        if len(words) >= 10:
            q_src.append(i); q_text.append(" ".join(words[:10]))
        if len(q_src) >= N_Q: break
    q_emb = model.encode(q_text, show_progress_bar=False)
    q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

    # Precompute per-query lexical top-pool (scores + ids) and vector top-pool.
    per_q = []
    for j, src in enumerate(q_src):
        bm = np.asarray(bm25.get_scores(tok(q_text[j])))
        cos = doc_emb @ q_emb[j]
        ltop = np.argsort(-bm)[:POOL]; dtop = np.argsort(-cos)[:POOL]
        per_q.append((bm, cos, ltop, dtop, src))

    # Build the offline NQC sketch from all queries' lexical top-pool scores.
    sample = np.array([nqc_cv(per_q[j][0][per_q[j][2]]) for j in range(len(per_q))])
    sample_sorted = np.sort(sample)
    def cdf(cv): return float(np.searchsorted(sample_sorted, cv, side="right")) / max(len(sample_sorted), 1)
    def dense_weight(cv, beta):
        if beta <= 0: return 1.0
        return min(max(1.0 - beta * cdf(cv), W_MIN), 1.0)

    print(f"[known-item] scifact corpus={len(docs)} queries={len(per_q)} POOL={POOL} K={K}")
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rec = mrr = 0.0
        for bm, cos, ltop, dtop, src in per_q:
            w = dense_weight(nqc_cv(bm[ltop]), beta)
            ln = dict(zip(ltop.tolist(), minmax(bm[ltop])))
            dn = dict(zip(dtop.tolist(), minmax(cos[dtop])))
            ids = set(ln) | set(dn)
            fused = {i: ln.get(i, 0.0) + w * dn.get(i, 0.0) for i in ids}
            ranked = sorted(fused, key=lambda i: -fused[i])
            if src in ranked[:K]:
                rank = ranked.index(src)
                rec += 1.0; mrr += 1.0 / (rank + 1)
        n = len(per_q)
        print(f"[pool-min-max NQC beta={beta:>4}] recall@{K}={rec/n:.4f}  MRR@{K}={mrr/n:.4f}")

if __name__ == "__main__":
    main()
