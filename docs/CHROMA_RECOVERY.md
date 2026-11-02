# Chroma snapshot recovery

## 2026-10-29

The deployed snapshot referenced 84,737 documents but its active HNSW segment
was missing `data_level0.bin`. The SQLite document store alone cannot reconstruct
all vectors: only 738 entries remained in its embeddings queue.

An intact snapshot in the original llama-bot project had identical document
metadata but different document IDs and a newer SQLite schema. It could not be
copied directly into the Chroma 1.0.20 deployment.

Recovery used `recover_chroma.py` to export the intact snapshot's existing
embeddings with its compatible Chroma environment, map them back to the original
IDs after comparing all metadata rows, and restore a new collection with Chroma
1.0.20. No embedding model or OpenAI calls were needed. The original L2 distance
metric was retained.

All 84,737 IDs, documents and metadata entries were compared in both directions
using SQLite EXCEPT queries, with zero differences. A new process reopened the
repaired snapshot with Chroma 1.0.20 and successfully queried its HNSW index.

The repaired snapshot is now `chroma_db_115/`. The original incomplete snapshot
is preserved at `chroma_db_115_backup_20261029/`. The source copy and intermediate
export remain in `chroma_db_115_recovered_20261029/` and
`chroma_db_115_recovery_work_20261029/`. These directories are ignored by Git and
excluded from the deployment allowlists. The source llama-bot database was not
modified.

## Deployment checks

`deploy.sh` runs `python3 validate_chroma.py` before submitting a build. It checks
the active collection's SQLite records and required persisted HNSW files, not
unreferenced segment directories. The Dockerfile additionally runs
`python validate_chroma.py --query` using the Linux deployment dependencies.
An index that cannot be loaded or queried fails the build before deployment.

These checks do not update an existing Cloud Run revision. A new image must be
built and deployed for the repaired snapshot to reach the live service.
