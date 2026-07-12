#!/usr/bin/env python
"""Perf lever refinement (dense-gating): which LABEL-FREE, DENSE-FREE signal best predicts
dense's per-query marginal value? Compare 3 signals, per corpus report dense marginal in the
low/high half (by signal median) + Pearson corr. Best gate = largest spread (one half ~0).
Signals (all from lexical retrieval + query, computable before running dense):
  clarity12 = (bm_top1 - bm_top2)/bm_top1     (last turn's)
  cv100     = std/mean of the top-100 BM25 scores  (NQC-style commitment)
  qlen      = number of stem+stop query terms       (short->dense helps more?)
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

SIGNALS = ["clarity12", "cv100", "qlen"]

def eval_corpus(ds, model, POOL=100):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    emb = model.encode(docs, show_progress_bar=False)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}
    sig = {s: [] for s in SIGNALS}; dmarg = []
    for q in q_ids:
        qtoks = tok(queries[q])
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = emb @ qe; bm = np.asarray(bm25.get_scores(qtoks))
        lorder = np.argsort(-bm); dtop = np.argsort(-cos)[:POOL]; ltop = lorder[:POOL]; rel = qrels[q]
        s1 = bm[lorder[0]]; s2 = bm[lorder[1]] if len(lorder) > 1 else 0.0
        top = bm[ltop]
        sig["clarity12"].append((s1 - s2)/(s1 + 1e-9) if s1 > 0 else 0.0)
        sig["cv100"].append(float(top.std()/(top.mean()+1e-9)))
        sig["qlen"].append(float(len(qtoks)))
        lex_nd = ndcg([idx2id[i] for i in ltop], rel)
        dn = dict(zip((idx2id[i] for i in dtop), minmax(cos[dtop])))
        ln = dict(zip((idx2id[i] for i in ltop), minmax(top)))
        ids = set(dn) | set(ln)
        fused = {d: dn.get(d,0.0) + ln.get(d,0.0) for d in ids}
        dmarg.append(ndcg(sorted(fused, key=lambda d: -fused[d]), rel) - lex_nd)
    return {s: np.array(sig[s]) for s in SIGNALS}, np.array(dmarg)

def main():
    datasets = sys.argv[1:] or ["scifact", "nfcorpus", "arguana", "scidocs"]
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    for ds in datasets:
        sig, dmarg = eval_corpus(ds, model)
        print(f"=== {ds} (q={len(dmarg)}, overall dmarg={dmarg.mean():+.4f}) ===")
        for s in SIGNALS:
            x = sig[s]; med = np.median(x)
            lo = dmarg[x <= med].mean(); hi = dmarg[x > med].mean()
            corr = float(np.corrcoef(x, dmarg)[0,1]) if x.std() > 0 and dmarg.std() > 0 else 0.0
            print(f"  {s:>9}: dmarg_low={lo:+.4f}  dmarg_high={hi:+.4f}  spread={abs(hi-lo):.4f}  corr={corr:+.3f}")
    print("[read] best gate = signal with largest |spread| where the low-dmarg half is ~0/negative")

if __name__ == "__main__":
    main()
