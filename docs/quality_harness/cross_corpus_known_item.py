#!/usr/bin/env python
"""Cross-corpus known-item NQC down-weight A/B: is the MRR gain robust across corpora, or
scifact-specific? Same known-item task (first ~10 words of held-out docs -> target) over the
4 BEIR corpora, pool-min-max fusion + NQC down-weight, MRR@10 vs beta.
"""
import json, os, re, sys
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

def load_corpus(ds):
    docs = []
    for line in open(os.path.join(HERE, ds, "corpus.jsonl")):
        o = json.loads(line)
        docs.append((o.get("title", "") + " " + o.get("text", "")).strip())
    return docs

def nqc_cv(scores):
    s = np.asarray(scores, dtype=np.float64); s = s[np.isfinite(s)]
    if len(s) == 0: return 0.0
    m = s.mean()
    return 0.0 if m <= 1e-10 else float(max(s.var(), 0.0) ** 0.5 / m)

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def eval_ds(ds, model, POOL=100, K=10, W_MIN=0.1, N_Q=400):
    docs = load_corpus(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    q_src, q_text = [], []
    for i, d in enumerate(docs):
        w = d.split()
        if len(w) >= 10:
            q_src.append(i); q_text.append(" ".join(w[:10]))
        if len(q_src) >= N_Q: break
    q_emb = model.encode(q_text, show_progress_bar=False)
    q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    per_q = []
    for j, src in enumerate(q_src):
        bm = np.asarray(bm25.get_scores(tok(q_text[j]))); cos = emb @ q_emb[j]
        per_q.append((bm, cos, np.argsort(-bm)[:POOL], np.argsort(-cos)[:POOL], src))
    ss = np.sort(np.array([nqc_cv(p[0][p[2]]) for p in per_q]))
    def dw(cv, beta):
        if beta <= 0: return 1.0
        pct = float(np.searchsorted(ss, cv, side="right")) / max(len(ss), 1)
        return min(max(1.0 - beta * pct, W_MIN), 1.0)
    out = {}
    for beta in BETAS:
        mrr = 0.0
        for bm, cos, ltop, dtop, src in per_q:
            w = dw(nqc_cv(bm[ltop]), beta)
            ln = dict(zip(ltop.tolist(), minmax(bm[ltop])))
            dn = dict(zip(dtop.tolist(), minmax(cos[dtop])))
            fused = {i: ln.get(i, 0.0) + w * dn.get(i, 0.0) for i in set(ln) | set(dn)}
            ranked = sorted(fused, key=lambda i: -fused[i])
            if src in ranked[:K]:
                mrr += 1.0 / (ranked.index(src) + 1)
        out[beta] = mrr / len(per_q)
    return len(per_q), out

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"{'corpus':>10} {'q':>4} | " + "  ".join(f"b={b}" for b in BETAS) + " | best_delta_vs_b0")
    wins = 0
    for ds in datasets:
        n, out = eval_ds(ds, model)
        best = max(out[b] for b in BETAS if b > 0) - out[0.0]
        if best > 0: wins += 1
        row = "  ".join(f"{out[b]:.4f}" for b in BETAS)
        print(f"{ds:>10} {n:>4} | {row} | {best:+.4f}")
    print(f"[summary] NQC down-weight improves MRR@10 in {wins}/{len(datasets)} corpora")

if __name__ == "__main__":
    main()
