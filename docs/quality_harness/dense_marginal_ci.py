#!/usr/bin/env python
"""Statistical rigor for the always-run-dense default: is the DENSE tier's marginal value
over lexical-alone statistically justified PER CORPUS, or a wash on some? Bootstrap 95% CI on
the per-query nDCG@10 delta (pool-min-max hybrid - lexical-only), stem+stop lexical +
model2vec dense, top-100 pool, per corpus + POOLED. 2000 resamples, seed=12345. Fresh:
dense_marginal.py reports single-run point estimates; this CIs the hybrid>lexical delta so
the corpus-dependent dense value (arguana ~0) can be read as decisive vs noise.
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

def per_query_deltas(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    deltas, hy_abs, lex_abs = [], [], []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
        ids = set(dn) | set(ln)
        hyb = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}          # pool-min-max hybrid
        lex = {idx2id[i]: bm[i] for i in ltop}                          # lexical-only
        n_h = ndcg(sorted(hyb, key=lambda d:-hyb[d]), rel)
        n_l = ndcg(sorted(lex, key=lambda d:-lex[d]), rel)
        deltas.append(n_h - n_l); hy_abs.append(n_h); lex_abs.append(n_l)
    return np.array(deltas), np.array(hy_abs), np.array(lex_abs)

def boot_ci(deltas, rng, n=2000):
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n)])
    return deltas.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    rng = np.random.default_rng(12345)
    pooled = []
    print("  corpus     q |  hybrid  lex   | dense_marginal(h-lex)   95% CI            CI>0")
    for ds in datasets:
        d, hy, lex = per_query_deltas(ds, model); pooled.append(d)
        m, lo, hi = boot_ci(d, rng)
        print(f"{ds:>10} {len(d):>5} | {hy.mean():.4f} {lex.mean():.4f} |   {m:+.4f}              [{lo:+.4f}, {hi:+.4f}]   {lo > 0}")
    alld = np.concatenate(pooled)
    m, lo, hi = boot_ci(alld, rng)
    print(f"{'POOLED':>10} {len(alld):>5} |                |   {m:+.4f}              [{lo:+.4f}, {hi:+.4f}]   {lo > 0}")

if __name__ == "__main__":
    main()
