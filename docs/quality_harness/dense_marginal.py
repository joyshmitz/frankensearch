#!/usr/bin/env python
"""Quality->perf: how much does the DENSE tier actually add to the hybrid on the corrected
(Tantivy-faithful stem+stop) baseline? The dense int8 scan is the dominant per-query CPU
cost; if its MARGINAL contribution (hybrid - lexical_alone) is ~0 on a corpus, dense is
skippable there (a real latency win). Report dense-alone / lexical-alone / hybrid(pool-min-
max), and each tier's marginal value, 4 corpora, nDCG@10, stem+stop lexical.
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
    nd = nl = nh = 0.0
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(tok(queries[q])))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        nd += ndcg([idx2id[i] for i in dtop], rel)
        nl += ndcg([idx2id[i] for i in ltop], rel)
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(bm[ltop])))
        ids = set(dn) | set(ln)
        fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        nh += ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
    n = len(q_ids)
    return n, nd/n, nl/n, nh/n

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"{'corpus':>10} {'q':>5} | {'dense':>7} {'lexical':>7} {'hybrid':>7} | {'dense_marg':>10} {'lex_marg':>9}")
    for ds in datasets:
        n, d, l, h = eval_corpus(ds, model)
        # marginal value of a tier = hybrid - (hybrid without that tier ~= other tier alone)
        dmarg = h - l   # what dense adds on top of lexical-alone
        lmarg = h - d   # what lexical adds on top of dense-alone
        print(f"{ds:>10} {n:>5} | {d:>7.4f} {l:>7.4f} {h:>7.4f} | {dmarg:>+10.4f} {lmarg:>+9.4f}")

if __name__ == "__main__":
    main()
