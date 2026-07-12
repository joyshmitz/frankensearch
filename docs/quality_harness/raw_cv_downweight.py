#!/usr/bin/env python
"""Deployment-faithfulness check for the NQC-adaptive dense down-weight win (9c1943df). The
win used in-set pctrank(cv) (mild leakage; production can't rank over the current query set).
Test a FIXED mapping instead: w_dense = clip(1 - beta*cv, w_min, 1) with RAW cv (cv = std/mean
of top-100 BM25, a self-normalizing statistic ~ corpus-independent scale) and the SAME beta on
all corpora. If it still wins 4/4, the lever is deployable with a fixed function (no query-sample
calibration). Also print each corpus's cv distribution. stem+stop, nDCG@10.
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

BETAS = [0.1, 0.2, 0.3, 0.4]  # w_dense = clip(1 - beta*cv, 0, 1); cv typically ~0.5-2

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    per_q = []; cvs = []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]; cv = float(top.std()/(top.mean()+1e-9))
        cvs.append(cv)
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        per_q.append((cv, dn, ln, rel))
    cvs = np.array(cvs)
    base = sum(ndcg(sorted({d: dn.get(d,0.0)+ln.get(d,0.0) for d in set(dn)|set(ln)},
                           key=lambda d:-(dn.get(d,0.0)+ln.get(d,0.0))), rel)
               for _, dn, ln, rel in per_q) / len(per_q)
    out = {}
    for beta in BETAS:
        tot = 0.0
        for cv, dn, ln, rel in per_q:
            w = max(0.0, min(1.0, 1.0 - beta*cv))
            ids = set(dn) | set(ln)
            fused = {d: w*dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
            tot += ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
        out[beta] = tot/len(per_q) - base
    return base, cvs, out

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    alld = {b: [] for b in BETAS}
    for ds in datasets:
        base, cvs, out = eval_corpus(ds, model)
        print(f"{ds:>10} base={base:.4f} cv[med={np.median(cvs):.2f} p10={np.percentile(cvs,10):.2f} p90={np.percentile(cvs,90):.2f}] | "
              + "  ".join(f"b={b}:{out[b]:+.4f}" for b in BETAS))
        for b in BETAS: alld[b].append(out[b])
    print("[robust] FIXED raw-cv mapping per-beta: min/mean across 4 corpora:")
    for b in BETAS:
        d = alld[b]
        print(f"  beta={b}: min={min(d):+.4f}  mean={np.mean(d):+.4f}  all>=-0.001={all(x>=-0.001 for x in d)}")

if __name__ == "__main__":
    main()
