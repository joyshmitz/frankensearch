#!/usr/bin/env python
"""Perf lever (dense-gating) — is a COMBINED signal a ROBUST (never-quality-negative across
corpora) gate? cv100-alone is Pareto on scifact/scidocs but negative on nfcorpus (the
landability blocker). Combine cv100 + clarity12 (both label-free/dense-free) via mean
percentile-rank; gate dense off on the top-f by the combined score. Report per-corpus hybrid
nDCG delta vs baseline at each f, and find the max f where ALL 4 corpora stay >= -0.001
(a robust conservative operating point). Compare to cv100-alone. stem+stop, nDCG@10.
"""
import json, os, re, math, sys
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

def load(ds):
    d = os.path.join(HERE, ds)
    docs, doc_ids = [], []
    for line in open(os.path.join(d, "corpus.jsonl")):
        o = json.loads(line); doc_ids.append(o["_id"])
        docs.append((o.get("title","") + " " + o.get("text","")).strip())
    queries = {}
    for line in open(os.path.join(d, "queries.jsonl")):
        o = json.loads(line); queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, dcid, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0: qrels.setdefault(q, {})[dcid] = int(rel)
    return doc_ids, docs, queries, [q for q in queries if q in qrels], qrels

def ndcg(ranked, rel, k=10):
    dcg = sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]) if rel.get(d,0))
    idcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(rel.values(),reverse=True)[:k]))
    return dcg/idcg if idcg else 0.0

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s-lo)/(hi-lo) if hi > lo else np.zeros_like(s)

def pctrank(x):
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)

FRACS = [0.0, 0.10, 0.25, 0.40, 0.50]

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    cv, clar, lexnd, hybnd = [], [], [], []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        lorder = np.argsort(-bm); dtop = np.argsort(-cos)[:POOL]; ltop = lorder[:POOL]; rel = qrels[q]
        top = bm[ltop]; s1 = bm[lorder[0]]; s2 = bm[lorder[1]] if len(lorder) > 1 else 0.0
        cv.append(float(top.std()/(top.mean()+1e-9)))
        clar.append((s1 - s2)/(s1 + 1e-9) if s1 > 0 else 0.0)
        lexnd.append(ndcg([idx2id[i] for i in ltop], rel))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        ids = set(dn) | set(ln)
        fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        hybnd.append(ndcg(sorted(fused, key=lambda d: -fused[d]), rel))
    return map(np.array, (cv, clar, lexnd, hybnd))

def gate_scores(sig, lex, hyb, n):
    order = np.argsort(-sig)
    out = []
    for f in FRACS:
        k = int(round(f*n)); skip = set(order[:k].tolist())
        out.append(np.array([lex[i] if i in skip else hyb[i] for i in range(n)]).mean())
    return out

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    combined_deltas = {f: [] for f in FRACS}
    for ds in datasets:
        cv, clar, lex, hyb = eval_corpus(ds, model); n = len(hyb); base = hyb.mean()
        combined = 0.5*pctrank(cv) + 0.5*pctrank(clar)
        cur_cv = gate_scores(cv, lex, hyb, n)
        cur_cm = gate_scores(combined, lex, hyb, n)
        print(f"=== {ds} (q={n}) base={base:.4f} ===")
        print("  cv100 :  " + "  ".join(f"f={int(f*100):>2}%:{s-base:+.4f}" for f, s in zip(FRACS, cur_cv)))
        print("  combo :  " + "  ".join(f"f={int(f*100):>2}%:{s-base:+.4f}" for f, s in zip(FRACS, cur_cm)))
        for f, s in zip(FRACS, cur_cm): combined_deltas[f].append(s - base)
    print("[robust] combined-gate per-f: min delta across 4 corpora (>=~0 = safe to gate that %):")
    for f in FRACS:
        ds_deltas = combined_deltas[f]
        print(f"  f={int(f*100):>2}%: min={min(ds_deltas):+.4f}  mean={np.mean(ds_deltas):+.4f}  all>=-0.001={all(d>=-0.001 for d in ds_deltas)}")

if __name__ == "__main__":
    main()
