#!/usr/bin/env python
"""Perf lever (adaptive dense-gating): is there a LABEL-FREE, DENSE-FREE per-query signal
that predicts when the dense scan won't help the hybrid? The gate must decide BEFORE running
dense, so the signal comes from LEXICAL retrieval + query only. Test: does BM25 clarity
(top1-top2 relative margin) predict dense's per-query marginal value (hybrid - lexical_alone)?
Bin queries low/high on the signal (per-corpus median), compare dense's mean marginal per bin.
If dense marginal is much LOWER in the high-clarity bin -> gate dense off there (skip the scan).
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

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    rows = []  # (clarity_signal, dense_marginal_for_this_query)
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(tok(queries[q])))
        lorder = np.argsort(-bm); dtop = np.argsort(-cos)[:POOL]; ltop = lorder[:POOL]
        rel = qrels[q]
        # LABEL-FREE, DENSE-FREE clarity signal: relative top1-top2 BM25 margin.
        s1 = bm[lorder[0]]; s2 = bm[lorder[1]] if len(lorder) > 1 else 0.0
        clarity = (s1 - s2) / (s1 + 1e-9) if s1 > 0 else 0.0
        lex_nd = ndcg([idx2id[i] for i in ltop], rel)
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
        ids = set(dn) | set(ln)
        fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        hyb_nd = ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
        rows.append((clarity, hyb_nd - lex_nd))
    return rows

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"{'corpus':>10} {'q':>5} | {'dmarg_lowClar':>13} {'dmarg_highClar':>14} | {'corr(clar,dmarg)':>16}")
    for ds in datasets:
        rows = eval_corpus(ds, model)
        clar = np.array([r[0] for r in rows]); dmarg = np.array([r[1] for r in rows])
        med = np.median(clar)
        lo = dmarg[clar <= med].mean(); hi = dmarg[clar > med].mean()
        # Pearson correlation between clarity and dense marginal
        corr = float(np.corrcoef(clar, dmarg)[0, 1]) if clar.std() > 0 and dmarg.std() > 0 else 0.0
        print(f"{ds:>10} {len(rows):>5} | {lo:>+13.4f} {hi:>+14.4f} | {corr:>+16.3f}")
    print("[read] gating works iff high-clarity dmarg << low-clarity dmarg AND corr is strongly negative")

if __name__ == "__main__":
    main()
