"""Transfer intact embeddings between Chroma versions while retaining original IDs.

Run export with the source-compatible Python environment, then restore with the
deployment-compatible environment. All destinations must be new paths.
"""

import argparse
import json
from pathlib import Path
import sqlite3

import chromadb
import numpy as np


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def export_vectors(source, original, archive, name):
    if archive.exists():
        raise ValueError("Archive already exists; refusing to overwrite it")
    with readonly(source / "chroma.sqlite3") as src, readonly(original / "chroma.sqlite3") as old:
        source_rows = dict(src.execute("SELECT embedding_id, id FROM embeddings"))
        original_rows = dict(old.execute("SELECT id, embedding_id FROM embeddings"))
        # Internal row IDs may be used only after verifying all document metadata.
        for query in (
            "SELECT id,key,string_value,int_value,float_value,bool_value "
            "FROM embedding_metadata ORDER BY id,key",
        ):
            if src.execute(query).fetchall() != old.execute(query).fetchall():
                raise ValueError("Source documents/metadata differ from the original snapshot")
        id_map = {source_id: original_rows[row] for source_id, row in source_rows.items()}
    collection = chromadb.PersistentClient(path=str(source)).get_collection(name)
    count = collection.count()
    if count != len(id_map):
        raise ValueError("Source collection count differs from the original snapshot")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive) as out:
        out.execute("CREATE TABLE records (id TEXT PRIMARY KEY, document TEXT, metadata TEXT, vector BLOB)")
        for offset in range(0, count, 1000):
            batch = collection.get(offset=offset, limit=1000, include=["documents", "metadatas", "embeddings"])
            out.executemany(
                "INSERT INTO records VALUES (?,?,?,?)",
                [(id_map[item_id], doc, json.dumps(meta), np.asarray(vec, dtype=np.float32).tobytes())
                 for item_id, doc, meta, vec in zip(batch["ids"], batch["documents"], batch["metadatas"], batch["embeddings"])],
            )
        actual = out.execute("SELECT count(*) FROM records").fetchone()[0]
        if actual != count:
            raise ValueError("Incomplete vector export")
    print(f"Exported {count} embeddings with original IDs", flush=True)


def restore_vectors(archive, target, original, name):
    if target.exists():
        raise ValueError("Target exists; refusing to overwrite it")
    with readonly(original / "chroma.sqlite3") as old:
        row = old.execute("SELECT config_json_str FROM collections WHERE name=?", (name,)).fetchone()
        # Preserve the original distance metric, rather than relying on defaults.
        config = json.loads(row[0])
        hnsw = config.get("vector_index", {}).get("hnsw") or config["hnsw_configuration"]
        space = hnsw["space"]
    collection = chromadb.PersistentClient(path=str(target)).create_collection(
        name, metadata={"hnsw:space": space}
    )
    total = 0
    with readonly(archive) as conn:
        cursor = conn.execute("SELECT id,document,metadata,vector FROM records ORDER BY id")
        while rows := cursor.fetchmany(1000):
            collection.add(
                ids=[row[0] for row in rows], documents=[row[1] for row in rows],
                metadatas=[json.loads(row[2]) for row in rows],
                embeddings=[np.frombuffer(row[3], dtype=np.float32).tolist() for row in rows],
            )
            total += len(rows)
            if total % 10000 == 0:
                print(f"Restored {total} embeddings", flush=True)
    if collection.count() != total:
        raise ValueError("Restored collection count mismatch")
    print(f"Restored {total} embeddings; Chroma={chromadb.__version__}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "restore"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--collection", default="rag_knowledge")
    args = parser.parse_args()
    if args.mode == "export":
        if args.source is None:
            parser.error("export requires --source")
        export_vectors(args.source, args.original, args.archive, args.collection)
    else:
        if args.target is None:
            parser.error("restore requires --target")
        restore_vectors(args.archive, args.target, args.original, args.collection)


if __name__ == "__main__":
    main()
