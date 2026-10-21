import sys, datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

q = " ".join(sys.argv[1:])
model = SentenceTransformer("all-MiniLM-L6-v2")
qc = QdrantClient(path="data/qdrant")
hits = qc.query_points("events", query=model.encode(q, normalize_embeddings=True).tolist(), limit=5).points
hits = [h for h in hits if h.score >= 0.25]
if not hits:
    print("no confident match")
for h in hits:
    p = h.payload
    print(f"{h.score:.2f} | {datetime.datetime.fromtimestamp(p['ts']):%Y-%m-%d %H:%M:%S} | {p['rule_id']} | {p['caption']}")
if hits:
    newest = max(h.payload["ts"] for h in hits)
    print(f"\nlast match: {datetime.datetime.fromtimestamp(newest):%Y-%m-%d %H:%M:%S}")
