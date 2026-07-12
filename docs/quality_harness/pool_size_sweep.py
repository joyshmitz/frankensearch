#!/usr/bin/env python
"""Quality: is the fusion candidate-pool size (engine `candidate_pool_size`, default ~100)
optimal? Sweep pool in {20,50,100,200,500} for pool-min-max on the stem+stop baseline, 4
corpora, nDCG@10. Embed once per corpus; all pool sizes evaluated in-memory from the full
argsort order (cheap). A robust non-100 winner would be a landable config change.
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

POOLS = [20, 50, 100, 200, 500]

def eval_corpus(ds, model):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    tot = {p: 0.0 for p in POOLS}
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(tok(queries[q])))
        dorder = np.argsort(-cos); lorder = np.argsort(-bm); rel = qrels[q]
        for p in POOLS:
            dtop, ltop = dorder[:p], lorder[:p]
            dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
            ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
            ids = set(dn) | set(ln)
            fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
            tot[p] += ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
    n = len(q_ids)
    return n, {p: tot[p]/n for p in POOLS}

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    hdr = "  ".join(f"p={p:>4}" for p in POOLS)
    print(f"{'corpus':>10} {'q':>5} | {hdr} | best")
    best_counts = {p: 0 for p in POOLS}
    for ds in datasets:
        n, scores = eval_corpus(ds, model)
        best = max(POOLS, key=lambda p: scores[p]); best_counts[best] += 1
        row = "  ".join(f"{scores[p]:.4f}" for p in POOLS)
        print(f"{ds:>10} {n:>5} | {row} | {best}")
    print(f"[summary] best-pool counts: " + ", ".join(f"{p}:{best_counts[p]}" for p in POOLS))

if __name__ == "__main__":
    main()
