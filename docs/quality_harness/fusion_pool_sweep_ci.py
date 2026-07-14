#!/usr/bin/env python
"""Robustness of the pool-min-max > RRF fusion finding to POOL SIZE (the latency-relevant
knob: a smaller candidate pool = less dense scan). fusion_pmm_vs_rrf_ci.py measured POOL=100;
does the advantage hold at small pools or is it a top-100 artifact? Encodes each corpus ONCE,
then evaluates the per-query nDCG@10 delta (pool-min-max - RRF) at POOL in {20, 50, 100} and
bootstrap-CIs the POOLED delta at each. stem+stop + model2vec, 2000 resamples, seed=12345.
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
        docs.append((o.get("title", "") + " " + o.get("text", "")).strip())
    queries = {}
    for line in open(os.path.join(d, "queries.jsonl")):
        o = json.loads(line); queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, dcid, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(q, {})[dcid] = int(rel)
    return doc_ids, docs, queries, [q for q in queries if q in qrels], qrels

def ndcg(ranked, rel, k=10):
    dcg = sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]) if rel.get(d,0))
    idcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(rel.values(),reverse=True)[:k]))
    return dcg/idcg if idcg else 0.0

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s-lo)/(hi-lo) if hi > lo else np.zeros_like(s)

RRF_K = 60; BIG = 10**9
POOLS = [20, 50, 100]

def per_corpus(ds, model):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    deltas = {P: [] for P in POOLS}
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks)); rel = qrels[q]
        dorder = np.argsort(-cos); lorder = np.argsort(-bm)
        for P in POOLS:
            dtop = dorder[:P]; ltop = lorder[:P]
            dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
            ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
            drank = {idx2id[i]: r for r, i in enumerate(dtop)}
            lrank = {idx2id[i]: r for r, i in enumerate(ltop)}
            ids = set(dn) | set(ln)
            mm = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
            rrf = {d: 1.0/(RRF_K+drank.get(d,BIG)) + 1.0/(RRF_K+lrank.get(d,BIG)) for d in ids}
            deltas[P].append(ndcg(sorted(mm, key=lambda d:-mm[d]), rel) - ndcg(sorted(rrf, key=lambda d:-rrf[d]), rel))
    return {P: np.array(v) for P, v in deltas.items()}

def boot_ci(deltas, rng, n=2000):
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n)])
    return deltas.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    rng = np.random.default_rng(12345)
    pooled = {P: [] for P in POOLS}
    for ds in datasets:
        pc = per_corpus(ds, model)
        for P in POOLS:
            pooled[P].append(pc[P])
    print("POOLED pool-min-max - RRF, by pool size (n=%d queries):" % sum(len(x) for x in pooled[POOLS[0]]))
    print("  POOL | mean_delta   95% CI            CI>0")
    for P in POOLS:
        alld = np.concatenate(pooled[P]); m, lo, hi = boot_ci(alld, rng)
        print(f"  {P:>4} | {m:+.4f}      [{lo:+.4f}, {hi:+.4f}]   {lo > 0}")

if __name__ == "__main__":
    main()
