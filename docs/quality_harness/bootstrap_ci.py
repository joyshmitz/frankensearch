#!/usr/bin/env python
"""Statistical rigor for the dense-downweight win (9c1943df / 910f0079): bootstrap 95% CI on
the per-query nDCG delta (down-weight - equal-weight baseline). If the CI excludes 0, the win
is statistically confirmed (not single-run noise). Down-weight = w_dense=1-0.5*pctrank(cv100),
stem+stop, per corpus + POOLED across all 4. 2000 bootstrap resamples over queries, seed=12345.
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
    return np.argsort(np.argsort(x)) / max(len(x)-1, 1)

BETA = 0.5

def per_query_deltas(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    recs = []; cvs = []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]; cvs.append(float(top.std()/(top.mean()+1e-9)))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        recs.append((dn, ln, rel))
    pr = pctrank(np.array(cvs))
    deltas = []
    for (dn, ln, rel), p in zip(recs, pr):
        ids = set(dn) | set(ln)
        b = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        w = 1.0 - BETA*p
        a = {d: w*dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        deltas.append(ndcg(sorted(a, key=lambda d:-a[d]), rel) - ndcg(sorted(b, key=lambda d:-b[d]), rel))
    return np.array(deltas)

def boot_ci(deltas, rng, n=2000):
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n)])
    return deltas.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    rng = np.random.default_rng(12345)
    pooled = []
    print("  corpus     q |   mean_delta   95% CI            CI>0")
    for ds in datasets:
        d = per_query_deltas(ds, model); pooled.append(d)
        m, lo, hi = boot_ci(d, rng)
        print(f"{ds:>10} {len(d):>5} |   {m:+.4f}     [{lo:+.4f}, {hi:+.4f}]   {lo > 0}")
    alld = np.concatenate(pooled)
    m, lo, hi = boot_ci(alld, rng)
    print(f"{'POOLED':>10} {len(alld):>5} |   {m:+.4f}     [{lo:+.4f}, {hi:+.4f}]   {lo > 0}")

if __name__ == "__main__":
    main()
