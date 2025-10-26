import argparse
import json
import hashlib
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from embedding import load_text_embedder
from utils import get_logger

DEFAULT_COLLECTION = "questdb_docs"
logger = get_logger("ingest")


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool = False) -> None:
    exist = client.get_collections()
    names = {c.name for c in exist.collections}
    if collection in names and recreate:
        logger.info(f"Recreating existing collection '{collection}'")
        client.delete_collection(collection)
        names.remove(collection)
    if collection not in names:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        logger.info(f"Created collection '{collection}' with vector size {vector_size}")


def make_point_id(source_url: str, section_title: str, chunk_text: str) -> int:
    h = hashlib.sha256()
    h.update(source_url.encode("utf-8"))
    h.update(b"|")
    h.update(section_title.encode("utf-8"))
    h.update(b"|")
    h.update(chunk_text.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


def batch(iterable: List[dict], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Embed and upload QuestDB docs to Qdrant")
    parser.add_argument("--jsonl", default="data/questdb_docs.jsonl")
    parser.add_argument("--host", default="http://localhost:6333")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    args = parser.parse_args()

    embedder = load_text_embedder(args.model)
    vector_size = embedder.get_sentence_embedding_dimension()
    client = QdrantClient(url=args.host)
    ensure_collection(client, args.collection, vector_size, recreate=args.recreate)

    records = list(read_jsonl(args.jsonl))
    points = []

    for rec in records:
        vec = embedder.encode([rec["text"]])[0]
        payload = {
            "source_url": rec.get("source_url"),
            "title": rec.get("title"),
            "section_title": rec.get("section"),
            "chunk_text": rec.get("text"),
        }
        pid = make_point_id(payload["source_url"], payload["section_title"], payload["chunk_text"])
        points.append(qmodels.PointStruct(id=pid, vector=vec.tolist(), payload=payload))

    for i, b in enumerate(batch(points, args.batch_size), start=1):
        client.upsert(collection_name=args.collection, points=b)
        logger.info(f"Upserted batch {i} ({len(b)} points)")

    logger.info(f"Ingested {len(points)} points into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
