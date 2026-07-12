#!/usr/bin/env python
"""Closes the dense-adaptive-fusion question: does a SOFT NQC-based dense down-weight (always
run dense, weight it LESS on high-commitment queries) achieve a ROBUST >=0 quality gain across
corpora where the HARD gate could not? fused = w_dense(cv)*dense_norm + lex_norm, with
w_dense = 1 - beta*pctrank(cv100). beta=0 is the baseline (pool-min-max, equal weight). If some
beta>0 is >=~0 on ALL 4 corpora -> a landable label-free quality win; else the arc is closed.
stem+stop, 4 corpora, nDCG@10.
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

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    per_q = []  # (cv, dense_norm_map, lex_norm_map, rel)
    cvs = []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]
        cvs.append(float(top.std()/(top.mean()+1e-9)))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        per_q.append((dn, ln, rel))
    cvs = np.array(cvs); pr = pctrank(cvs)
    scores = {}
    for beta in BETAS:
        tot = 0.0
        for (dn, ln, rel), p in zip(per_q, pr):
            w = 1.0 - beta*p
            ids = set(dn) | set(ln)
            fused = {d: w*dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
            tot += ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
        scores[beta] = tot/len(per_q)
    return scores

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    all_deltas = {b: [] for b in BETAS}
    for ds in datasets:
        sc = eval_corpus(ds, model); base = sc[0.0]
        print(f"{ds:>10} base={base:.4f} | " + "  ".join(f"b={b}:{sc[b]-base:+.4f}" for b in BETAS if b > 0))
        for b in BETAS: all_deltas[b].append(sc[b]-base)
    print("[robust] soft-downweight per-beta: min/mean delta across 4 corpora:")
    for b in BETAS:
        if b == 0: continue
        d = all_deltas[b]
        print(f"  beta={b}: min={min(d):+.4f}  mean={np.mean(d):+.4f}  all>=-0.001={all(x>=-0.001 for x in d)}")

if __name__ == "__main__":
    main()
