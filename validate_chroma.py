"""Check the bundled Chroma snapshot without modifying its SQLite database."""

import argparse
from pathlib import Path
import sqlite3


def validate_snapshot(db_dir: Path, collection_name: str) -> int:
    database = db_dir / "chroma.sqlite3"
    if not database.is_file():
        raise ValueError(f"Missing Chroma database: {database}")

    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        collection = conn.execute(
            "SELECT id FROM collections WHERE name = ?", (collection_name,)
        ).fetchone()
        if collection is None:
            raise ValueError(f"Missing Chroma collection: {collection_name}")
        segments = conn.execute(
            "SELECT id FROM segments WHERE collection = ? AND type = ?",
            (collection[0], "urn:chroma:segment/vector/hnsw-local-persisted"),
        ).fetchall()
        count = conn.execute(
            "SELECT count(*) FROM embeddings e JOIN segments s ON e.segment_id = s.id "
            "WHERE s.collection = ?", (collection[0],)
        ).fetchone()[0]

    if not count or not segments:
        raise ValueError("The RAG snapshot has no documents or persisted vector segment")
    for (segment_id,) in segments:
        for name in ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin",
                     "index_metadata.pickle"):
            path = db_dir / segment_id / name
            if not path.is_file():
                raise ValueError(f"Incomplete Chroma snapshot; missing {path}")
            if name in ("header.bin", "data_level0.bin") and path.stat().st_size == 0:
                raise ValueError(f"Empty Chroma index file: {path}")
    return count


def probe_query(db_dir: Path, collection_name: str, expected_count: int) -> None:
    import chromadb

    collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(collection_name)
    if collection.count() != expected_count:
        raise ValueError("Chroma collection count does not match SQLite document count")
    sample = collection.get(limit=1, include=["embeddings"])
    if not sample["ids"] or sample["embeddings"] is None:
        raise ValueError("Cannot read a sample embedding from the Chroma index")
    result = collection.query(
        query_embeddings=sample["embeddings"].tolist(), n_results=1, include=["distances"]
    )
    if not result["ids"] or not result["ids"][0]:
        raise ValueError("Chroma index query returned no results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=Path(__file__).parent / "chroma_db_115")
    parser.add_argument("--collection", default="rag_knowledge")
    parser.add_argument("--query", action="store_true", help="Also load and query the HNSW index")
    args = parser.parse_args()
    try:
        count = validate_snapshot(args.db_dir, args.collection)
        if args.query:
            probe_query(args.db_dir, args.collection, count)
    except (ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"Chroma validation failed: {exc}\n")
    print(f"Chroma snapshot validated: collection={args.collection}, documents={count}")


if __name__ == "__main__":
    main()
