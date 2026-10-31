"""
query_vector_db.py
-------------------
Queries the Chroma vector database built by build_vector_db.py.

In production, a user types a question -> you embed it with the SAME
model used for the chunks (bge-m3) -> pass that vector here.

This script has two modes:

1. embed_and_query(question)
   The real path: embeds a live question with bge-m3, then searches.
   Requires the 'FlagEmbedding' or 'sentence-transformers' package and
   an internet connection to download the bge-m3 model weights the
   first time it runs.

2. query_by_existing_chunk(chunk_id)
   A demo/debugging path that needs NO model download: it reuses an
   embedding already stored in the DB as a stand-in "query", so you can
   confirm the retrieval mechanics work end-to-end before wiring up
   live embedding.
"""

import json
import chromadb
from tempCodeRunnerFile import question

DB_PATH = "./chroma_db"
COLLECTION_NAME = "ain_chunks"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def print_results(results):
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (id_, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        similarity = 1 - dist  # chroma returns distance; smaller = closer
        print(f"\n#{rank}  chunk_id: {id_}   (similarity ≈ {similarity:.3f})")
        print(f"    chapter {meta['chapter']}, clause {meta['clause_number']}")
        print(f"    title: {meta['title']}")
        print(f"    summary: {meta['summary']}")
        print(f"    content: {doc[:150]}{'...' if len(doc) > 150 else ''}")


def query_by_existing_chunk(chunk_id, top_k=5):
    """
    Demo mode: no live embedding needed. Uses a chunk already in the DB
    as the 'query', and searches for what's semantically closest to it
    (including itself, which should always come back as the #1 match
    with similarity ≈ 1.0 — a good sanity check that the DB is wired up
    correctly).
    """
    collection = get_collection()
    existing = collection.get(ids=[chunk_id], include=["embeddings"])

    if not existing["ids"]:
        print(f"chunk_id '{chunk_id}' not found in the collection.")
        return

    query_vector = existing["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    print(f"Query: using embedding of chunk '{chunk_id}' as a stand-in query")
    print_results(results)


def embed_and_query(question, top_k=5):
    """
    Real mode: embeds a live natural-language question with bge-m3,
    then searches the vector DB.

    NOTE: requires `pip install FlagEmbedding` (or sentence-transformers)
    and downloads ~2GB of model weights on first run from HuggingFace.
    """
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    query_vector = model.encode([question])["dense_vecs"][0].tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    print(f"Query: {question}")
    print_results(results)


if __name__ == "__main__":
    # Demo run — no internet/model download required.
    # Picks a real chunk_id from ch1 and shows it retrieves itself + neighbors.
    query_by_existing_chunk("ain_ch1_१", top_k=5)

    print("\n" + "=" * 70)
    print("To run a REAL query with a live question, uncomment below")
    print("(requires bge-m3 model download):")
    print("=" * 70)
    embed_and_query(question, top_k=5)