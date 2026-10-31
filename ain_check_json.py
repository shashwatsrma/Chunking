import json
from collections import Counter

FILE = "chunks_with_embeddings.json" 
EXPECTED_KEYS = {"chunk_id", "chapter", "clause_number", "title",
                 "summary", "content_type", "content", "embedding"}

records = json.load(open(FILE, encoding="utf-8"))
ids = [r.get("chunk_id") for r in records]

print("records:", len(records), "| unique ids:", len(set(ids)))
print("duplicate ids:", [k for k, c in Counter(ids).items() if c > 1][:10])
print("id prefixes:", Counter(str(i).split("_")[0] for i in ids))
print("missing keys:", Counter(k for r in records for k in EXPECTED_KEYS - set(r)))
print("extra keys:", Counter(k for r in records for k in set(r) - EXPECTED_KEYS))
print("embedding dims:", Counter(len(r["embedding"]) if r.get("embedding") else 0 for r in records))
print("chapter types:", Counter(type(r.get("chapter")).__name__ for r in records))
print("clause_number types:", Counter(type(r.get("clause_number")).__name__ for r in records))
print("content_type:", Counter(r.get("content_type") for r in records))
print("empty content:", sum(1 for r in records if not r.get("content")))
print("empty title:", sum(1 for r in records if not r.get("title")))
print("empty content AND title:", sum(1 for r in records if not r.get("content") and not r.get("title")))