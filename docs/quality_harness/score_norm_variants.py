#!/usr/bin/env python
"""Quality: does z-score score-normalization beat min-max for pool fusion? The shipped
pool-min-max kernel (a9e53b4) uses min-max [0,1]. z-score (standardize) preserves relative
spacing differently. If z-score robustly beats min-max cross-corpus, it's a landable change
to the fusion normalization. stem+stop baseline, 4 BEIR corpora, nDCG@10. Absent-from-pool
docs get that tier's pool MINIMUM normalized score (consistent across both variants).
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

def norm_minmax(s):
    lo, hi = s.min(), s.max()
    n = (s-lo)/(hi-lo) if hi > lo else np.zeros_like(s)
    return n, 0.0  # (normalized, absent-fill = pool min = 0)

def norm_zscore(s):
    mu, sd = s.mean(), s.std()
    n = (s-mu)/sd if sd > 0 else np.zeros_like(s)
    return n, (n.min() if len(n) else 0.0)  # absent-fill = pool min z

def fuse_eval(ds, model, normfn, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    tot = 0.0
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(tok(queries[q])))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]
        dn, dfill = normfn(cos[dtop]); ln, lfill = normfn(bm[ltop])
        dmap = dict(zip((idx2id[i] for i in dtop), dn))
        lmap = dict(zip((idx2id[i] for i in ltop), ln))
        ids = set(dmap) | set(lmap)
        fused = {d: dmap.get(d, dfill) + lmap.get(d, lfill) for d in ids}
        tot += ndcg(sorted(fused, key=lambda d: -fused[d]), qrels[q])
    return len(q_ids), tot/len(q_ids)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"{'corpus':>10} {'q':>5} | {'minmax':>8} {'zscore':>8} | {'z-mm':>8}")
    zwins = 0
    for ds in datasets:
        _, mm = fuse_eval(ds, model, norm_minmax)
        n, zs = fuse_eval(ds, model, norm_zscore)
        if zs > mm: zwins += 1
        print(f"{ds:>10} {n:>5} | {mm:>8.4f} {zs:>8.4f} | {zs-mm:>+8.4f}")
    print(f"[summary] z-score > min-max in {zwins}/{len(datasets)} corpora")

if __name__ == "__main__":
    main()
