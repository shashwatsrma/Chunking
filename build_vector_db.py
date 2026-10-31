"""
build_vector_db.py
-------------------
Builds a persistent Chroma vector database from the Ain (2049)
chunks + their bge-m3 embeddings.

WHAT THIS DOES:
1. Loads chunks_with_embeddings.json — a single file where each chunk record
   already contains its text, metadata, AND its 1024-dim bge-m3 embedding
   together (no need to join two separate files anymore).
2. Writes everything into a Chroma collection stored on disk at ./chroma_db

Run once to build the DB:
    python build_vector_db.py

After this runs, ./chroma_db/ is your "database" — a folder on disk.
Re-running this script will overwrite the collection with fresh data
(safe to re-run after regenerating embeddings).
"""

import json
import chromadb

DB_PATH = "./chroma_db"
COLLECTION_NAME = "ain_chunks"  # COLLECTION_NAME: a collection is like a table in a normal database. It’s a named group of chunks. One database folder can hold many collections.
SOURCE_DOCUMENT = "ain_2049"  # used for filtering later
CHUNKS_FILE = "chunks_with_embeddings.json"


def load_chunks_with_embeddings():
    """Load all chunk records — each one already has its text, metadata,
    and embedding bundled together in one file."""
    records = json.load(open(CHUNKS_FILE, encoding="utf-8"))
    print(f"JSON records: {len(records)}")
    return {r["chunk_id"]: r for r in records} # dictionary 
   

def build():
    chunks = load_chunks_with_embeddings()

    print(f"Loaded {len(chunks)} chunks (with embeddings already attached).")

    # Sanity check: every chunk should actually have an embedding field
    missing = [cid for cid, c in chunks.items() if not c.get("embedding")]
    if missing:
        print(f"WARNING: {len(missing)} chunks have no embedding, skipping them:")
        for m in missing[:5]:
            print("  -", m)

    # Create a persistent client — this is what writes to disk at DB_PATH
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete any existing collection with the same name so re-runs start clean
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Motor Vehicles and Transport Management Act 2049 (chapters 1-11) ",
            "hnsw:space": "cosine",  # IMPORTANT: Chroma defaults to L2 distance, not
            # cosine. Our embeddings (bge-m3) are compared meaningfully with cosine
            # similarity, so this must be set explicitly or "similarity" scores
            # printed later will be wrong for anything that isn't a near-exact match.
        }
    )

    ids = []
    vectors = []
    documents = []
    metadatas = []

    for chunk_id, chunk in chunks.items():
        if not chunk.get("embedding"):
            continue

        text = chunk.get("content") or chunk.get("title") or ""

        ids.append(chunk_id)
        vectors.append(chunk["embedding"])
        documents.append(text)
        metadatas.append({
            "chapter": chunk.get("chapter") or 0,
            "clause_number": chunk.get("clause_number") or "",
            "title": chunk.get("title") or "",
            "summary": chunk.get("summary") or "",
            "content_type": chunk.get("content_type") or "",
            "source_document": SOURCE_DOCUMENT, # used for filtering later
        })

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Added {len(ids)} chunks to Chroma collection '{COLLECTION_NAME}'.")
    print(f"Database written to: {DB_PATH}")
    print(f"Total chunks now in collection: {collection.count()}")


if __name__ == "__main__":
    build()