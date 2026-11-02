from pathlib import Path
import sqlite3
import tempfile
import unittest

from validate_chroma import validate_snapshot


class ValidateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with sqlite3.connect(self.root / "chroma.sqlite3") as conn:
            conn.executescript(
                "CREATE TABLE collections (id TEXT, name TEXT);"
                "CREATE TABLE segments (id TEXT, collection TEXT, type TEXT);"
                "CREATE TABLE embeddings (id INTEGER, segment_id TEXT);"
                "INSERT INTO collections VALUES ('collection', 'rag_knowledge');"
                "INSERT INTO segments VALUES ('active', 'collection', "
                "'urn:chroma:segment/vector/hnsw-local-persisted');"
                "INSERT INTO embeddings VALUES (1, 'active');"
            )
        segment = self.root / "active"
        segment.mkdir()
        for name in ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin",
                     "index_metadata.pickle"):
            (segment / name).write_bytes(b"test")

    def test_complete_snapshot(self):
        self.assertEqual(validate_snapshot(self.root, "rag_knowledge"), 1)

    def test_missing_vector_file_fails(self):
        (self.root / "active" / "data_level0.bin").unlink()
        with self.assertRaisesRegex(ValueError, "data_level0.bin"):
            validate_snapshot(self.root, "rag_knowledge")

    def test_empty_vector_file_fails(self):
        (self.root / "active" / "data_level0.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty Chroma index"):
            validate_snapshot(self.root, "rag_knowledge")

    def test_missing_collection_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing Chroma collection"):
            validate_snapshot(self.root, "other")

    def test_missing_database_is_not_created(self):
        absent = self.root / "absent"
        with self.assertRaisesRegex(ValueError, "Missing Chroma database"):
            validate_snapshot(absent, "rag_knowledge")
        self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
