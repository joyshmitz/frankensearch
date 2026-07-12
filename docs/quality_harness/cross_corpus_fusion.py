#!/usr/bin/env python
"""Cross-corpus: RRF vs pool-min-max score fusion on the stem+stop baseline, over 4 BEIR
corpora. Tests whether last turn's scifact finding (pool-min-max beats RRF, margin grows
with stem+stop) is robust or scifact-specific. Also reports basic-tok RRF/mm for contrast.
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
def tok_basic(s): return TOKEN.findall(s.lower())
def tok_ss(s): return _stem.stemWords([t for t in TOKEN.findall(s.lower()) if t not in STOP])

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
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s-lo)/(hi-lo) if hi > lo else np.zeros_like(s)

def eval_corpus(ds, model, tok, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    n_rrf = n_mm = 0.0
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe
        bm = np.asarray(bm25.get_scores(tok(queries[q])))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]
        rel = qrels[q]; BIG = 10**9
        drank = {idx2id[i]: r for r, i in enumerate(dtop)}
        lrank = {idx2id[i]: r for r, i in enumerate(ltop)}
        ids = set(drank) | set(lrank)
        rrf = {d: 1.0/(60+drank.get(d,BIG)) + 1.0/(60+lrank.get(d,BIG)) for d in ids}
        n_rrf += ndcg(sorted(rrf, key=lambda d: -rrf[d]), rel)
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
        mm = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        n_mm += ndcg(sorted(mm, key=lambda d: -mm[d]), rel)
    n = len(q_ids)
    return n, n_rrf/n, n_mm/n

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"{'corpus':>10} {'q':>5} | {'RRF_basic':>9} {'mm_basic':>9} | {'RRF_ss':>8} {'mm_ss':>8} | {'mm-RRF(ss)':>10}")
    wins = 0
    for ds in datasets:
        _, rrf_b, mm_b = eval_corpus(ds, model, tok_basic)
        n, rrf_s, mm_s = eval_corpus(ds, model, tok_ss)
        delta = mm_s - rrf_s
        if delta > 0: wins += 1
        print(f"{ds:>10} {n:>5} | {rrf_b:>9.4f} {mm_b:>9.4f} | {rrf_s:>8.4f} {mm_s:>8.4f} | {delta:>+10.4f}")
    print(f"[summary] pool-min-max > RRF on stem+stop in {wins}/{len(datasets)} corpora")

if __name__ == "__main__":
    main()
