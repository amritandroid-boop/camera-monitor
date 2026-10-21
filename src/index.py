import sqlite3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
qc = QdrantClient(path="data/qdrant")
if not qc.collection_exists("events"):
    qc.create_collection("events", vectors_config=VectorParams(size=384, distance=Distance.COSINE))

db = sqlite3.connect("data/events.db")
rows = db.execute("select rowid, ts, rule_id, caption, thumb from events "
                  "where caption is not null and caption != '' and caption not like '%?' and caption not like '%triggered is true%'").fetchall()
vecs = model.encode([r[3] for r in rows], normalize_embeddings=True)
points = [PointStruct(id=r[0], vector=v.tolist(),
                      payload={"ts": r[1], "rule_id": r[2], "caption": r[3], "thumb": r[4]})
          for r, v in zip(rows, vecs)]
qc.upsert("events", points)
print(len(points), "events indexed")