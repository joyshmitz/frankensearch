#!/usr/bin/env python
"""Perf lever (dense-gating) — the DEPLOYABLE tradeoff curve. Gate dense off on the top-f
fraction of queries by cv100 (NQC, the best label-free/dense-free signal); measure hybrid
nDCG vs f (= fraction of dense scans SKIPPED). Anchors: f=0 (always dense, baseline), f=1
(never dense = lexical-only), and ORACLE (skip exactly the queries where dense doesn't help,
dmarg<=0) = the best any gate could do. The gap cv100 vs oracle = cost of the weak signal.
stem+stop baseline, 4 corpora, nDCG@10.
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

FRACS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    cv, lexnd, hybnd = [], [], []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]
        cv.append(float(top.std()/(top.mean()+1e-9)))
        lexnd.append(ndcg([idx2id[i] for i in ltop], rel))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        ids = set(dn) | set(ln)
        fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        hybnd.append(ndcg(sorted(fused, key=lambda d: -fused[d]), rel))
    return np.array(cv), np.array(lexnd), np.array(hybnd)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    for ds in datasets:
        cv, lex, hyb = eval_corpus(ds, model)
        n = len(hyb); dmarg = hyb - lex
        base = hyb.mean()
        # cv100-gate: skip dense (use lexical) on the top-f fraction by cv (highest commitment)
        order = np.argsort(-cv)  # descending cv
        print(f"=== {ds} (q={n}) baseline hybrid={base:.4f}  lexical-only={lex.mean():.4f} ===")
        cells = []
        for f in FRACS:
            k = int(round(f*n)); skip = set(order[:k].tolist())
            score = np.array([lex[i] if i in skip else hyb[i] for i in range(n)]).mean()
            cells.append(f"f={int(f*100):>3}%:{score:.4f}")
        print("  cv100-gate  " + "  ".join(cells))
        # oracle: skip dense on all queries where dense doesn't help (dmarg<=0)
        oracle_skip = (dmarg <= 0).mean()
        oracle = np.maximum(hyb, lex).mean()
        print(f"  ORACLE  skip={oracle_skip*100:.1f}%  hybrid={oracle:.4f}  (never let dense hurt; +{oracle-base:.4f} vs base)")
    print("[read] free-lunch = the f where cv100-gate score >= baseline (skip that % of dense scans at no/plus quality)")

if __name__ == "__main__":
    main()
