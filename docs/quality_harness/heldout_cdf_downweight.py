#!/usr/bin/env python
"""Final de-risk for the dense-downweight land: the TRUE deployment path. Build the cv CDF
from a HELD-OUT calibration half of queries; apply w_dense = 1 - beta*CDF_calib(cv) to the
DISJOINT eval half (unseen). If the win survives this leakage-free split, it is deployment-
validated (matches shipping `w=clip(1-beta*CDF(cv))` with an offline query-sample cv sketch).
stem+stop, 4 corpora, nDCG@10 on the eval half, delta vs eval baseline (beta=0).
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

BETAS = [0.5, 1.0]

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    recs = []  # (cv, dn, ln, rel)
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        dtop = np.argsort(-cos)[:POOL]; ltop = np.argsort(-bm)[:POOL]; rel = qrels[q]
        top = bm[ltop]; cv = float(top.std()/(top.mean()+1e-9))
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        recs.append((cv, dn, ln, rel))
    # interleaved split: calib = odd index, eval = even index (leakage-free)
    calib_cv = np.array([recs[i][0] for i in range(len(recs)) if i % 2 == 1])
    calib_sorted = np.sort(calib_cv)
    eval_recs = [recs[i] for i in range(len(recs)) if i % 2 == 0]
    def cdf(cv):  # fraction of calib cv <= cv
        return float(np.searchsorted(calib_sorted, cv, side="right")) / max(len(calib_sorted), 1)
    base = sum(ndcg(sorted({d: dn.get(d,0.0)+ln.get(d,0.0) for d in set(dn)|set(ln)},
                           key=lambda d:-(dn.get(d,0.0)+ln.get(d,0.0))), rel)
               for _, dn, ln, rel in eval_recs) / len(eval_recs)
    out = {}
    for beta in BETAS:
        tot = 0.0
        for cv, dn, ln, rel in eval_recs:
            w = max(0.0, min(1.0, 1.0 - beta*cdf(cv)))
            ids = set(dn) | set(ln)
            fused = {d: w*dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
            tot += ndcg(sorted(fused, key=lambda d: -fused[d]), rel)
        out[beta] = tot/len(eval_recs) - base
    return len(eval_recs), base, out

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    alld = {b: [] for b in BETAS}
    for ds in datasets:
        n, base, out = eval_corpus(ds, model)
        print(f"{ds:>10} eval_q={n} base={base:.4f} | " + "  ".join(f"b={b}:{out[b]:+.4f}" for b in BETAS))
        for b in BETAS: alld[b].append(out[b])
    print("[deployment-validated?] held-out-CDF down-weight, min/mean delta across 4 corpora:")
    for b in BETAS:
        d = alld[b]
        print(f"  beta={b}: min={min(d):+.4f}  mean={np.mean(d):+.4f}  all>=-0.001={all(x>=-0.001 for x in d)}")

if __name__ == "__main__":
    main()
