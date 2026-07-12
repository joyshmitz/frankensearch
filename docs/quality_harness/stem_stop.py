#!/usr/bin/env python
"""Quality experiment: Tantivy-faithful stem+stop lexical analysis vs basic tokenization.

The rebuilt harness's lexical tier uses basic whitespace+lowercase tokenization; the engine's
Tantivy analyzer applies Snowball(English) stemming + English stopword removal. The memory
documents stem+stop as the biggest quality lever (~+5.8% on scidocs). Embed scifact ONCE, build
TWO BM25 indexes (basic vs stem+stop), and compare dense/lexical/hybrid nDCG@10. Snowball English
== Tantivy's default stemmer; stopwords = the standard Lucene/Tantivy English set.
"""
import json, os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
from model2vec import StaticModel
from rank_bm25 import BM25Okapi
import snowballstemmer

TOKEN = re.compile(r"[a-z0-9]+")
# Tantivy/Lucene default English stopwords.
STOP = set("a an and are as at be but by for if in into is it no not of on or such "
           "that the their then there these they this to was will with".split())
_stemmer = snowballstemmer.stemmer("english")

def tok_basic(s):
    return TOKEN.findall(s.lower())

def tok_stem_stop(s):
    toks = [t for t in TOKEN.findall(s.lower()) if t not in STOP]
    return _stemmer.stemWords(toks)

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
    q_ids = [q for q in queries if q in qrels]
    return doc_ids, docs, queries, q_ids, qrels

def ndcg_at_k(ranked, rel, k=10):
    dcg = sum((2**rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if rel.get(d, 0))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def rrf(a, b, k=60):
    ids = set(a) | set(b); BIG = 10**9
    return {d: 1.0/(k+a.get(d, BIG)) + 1.0/(k+b.get(d, BIG)) for d in ids}

def main():
    ds = "scifact"
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    doc_emb = model.encode(docs, show_progress_bar=False)
    doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
    idx2id = {i: d for i, d in enumerate(doc_ids)}

    variants = {"basic": tok_basic, "stem_stop": tok_stem_stop}
    bm25 = {name: BM25Okapi([fn(d) for d in docs]) for name, fn in variants.items()}

    print(f"[ds] {ds}  q={len(q_ids)}  corpus={len(docs)}")
    for name, fn in variants.items():
        nd = nl = nh = 0.0
        for q in q_ids:
            qe = model.encode([queries[q]], show_progress_bar=False)[0]
            qe /= (np.linalg.norm(qe) + 1e-9)
            cos = doc_emb @ qe
            bm = np.asarray(bm25[name].get_scores(fn(queries[q])))
            dtop = np.argsort(-cos)[:100]; ltop = np.argsort(-bm)[:100]
            drank = {idx2id[i]: r for r, i in enumerate(dtop)}
            lrank = {idx2id[i]: r for r, i in enumerate(ltop)}
            rel = qrels[q]
            nd += ndcg_at_k([idx2id[i] for i in dtop], rel)
            nl += ndcg_at_k([idx2id[i] for i in ltop], rel)
            fused = rrf(drank, lrank)
            nh += ndcg_at_k(sorted(fused, key=lambda d: -fused[d]), rel)
        n = len(q_ids)
        print(f"[{name:>9}] dense={nd/n:.4f}  lexical={nl/n:.4f}  hybrid={nh/n:.4f}")

if __name__ == "__main__":
    main()
