#!/usr/bin/env python
"""Do the two shipped dense-reduction quality levers COMPOUND or OVERLAP? pool-min-max score
fusion (a9e53b4) and the NQC dense down-weight (ac081b7d) BOTH damp the dense tier's
over-contribution (pool-min-max via per-tier min-max normalization; down-weight via an explicit
per-query weight w=1-0.5*pctrank(cv100)). If they target the same failure they should be
SUB-ADDITIVE. Measures, per query, two bootstrap-CI'd nDCG@10 deltas, stem+stop + model2vec,
top-100 pool, per corpus + POOLED (2000 resamples, seed=12345):
  C = down-weighted-pool-min-max  -  RRF            (FULL shipped stack vs naive baseline)
  B = down-weighted-pool-min-max  -  pool-min-max   (down-weight increment on top of pmm)
Compare pooled C against A+B, where A = pool-min-max - RRF = +0.0041 (fusion_pmm_vs_rrf_ci.py).
Additive iff C ~= A + B; C < A + B => the levers overlap.
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

def pctrank(x):
    return np.argsort(np.argsort(x)) / max(len(x)-1, 1)

RRF_K = 60; BIG = 10**9; BETA = 0.5

def per_query(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    recs, cvs = [], []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]; cvs.append(float(top.std()/(top.mean()+1e-9)))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        drank = {idx2id[i]: r for r, i in enumerate(dtop)}
        lrank = {idx2id[i]: r for r, i in enumerate(ltop)}
        recs.append((dn, ln, drank, lrank, rel))
    pr = pctrank(np.array(cvs))
    dC, dB = [], []
    for (dn, ln, drank, lrank, rel), p in zip(recs, pr):
        ids = set(dn) | set(ln)
        w = 1.0 - BETA * p
        rrf = {d: 1.0/(RRF_K+drank.get(d,BIG)) + 1.0/(RRF_K+lrank.get(d,BIG)) for d in ids}
        pmm = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        dwp = {d: w*dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        n_rrf = ndcg(sorted(rrf, key=lambda d:-rrf[d]), rel)
        n_pmm = ndcg(sorted(pmm, key=lambda d:-pmm[d]), rel)
        n_dwp = ndcg(sorted(dwp, key=lambda d:-dwp[d]), rel)
        dC.append(n_dwp - n_rrf); dB.append(n_dwp - n_pmm)
    return np.array(dC), np.array(dB)

def boot_ci(deltas, rng, n=2000):
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n)])
    return deltas.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    rng = np.random.default_rng(12345)
    pC, pB = [], []
    print("  corpus     q | C=stack-RRF   95%CI          C>0 | B=dwt-incr   95%CI          B>0")
    for ds in datasets:
        c, b = per_query(ds, model); pC.append(c); pB.append(b)
        mc, loc, hic = boot_ci(c, rng); mb, lob, hib = boot_ci(b, rng)
        print(f"{ds:>10} {len(c):>5} | {mc:+.4f} [{loc:+.4f},{hic:+.4f}] {str(loc>0):>5} | {mb:+.4f} [{lob:+.4f},{hib:+.4f}] {str(lob>0):>5}")
    aC = np.concatenate(pC); aB = np.concatenate(pB)
    mc, loc, hic = boot_ci(aC, rng); mb, lob, hib = boot_ci(aB, rng)
    print(f"{'POOLED':>10} {len(aC):>5} | {mc:+.4f} [{loc:+.4f},{hic:+.4f}] {str(loc>0):>5} | {mb:+.4f} [{lob:+.4f},{hib:+.4f}] {str(lob>0):>5}")
    print(f"\nAdditivity check (pooled): A(fusion,prior)= +0.0041 ; B(down-weight)= {mb:+.4f} ; A+B= {0.0041+mb:+.4f} ; C(stack)= {mc:+.4f}")
    print(f"=> {'SUB-ADDITIVE (levers overlap)' if mc < 0.0041 + mb - 0.0005 else 'approx ADDITIVE'}")

if __name__ == "__main__":
    main()
