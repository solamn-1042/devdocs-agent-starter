"""Day5离线测试：FakeCollection不下载模型、不访问API或Chroma。"""
import unittest
from vector_store import chunk_key, fingerprint, sync_index, search_index


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def get(self, include=None):
        return {"ids": list(self.rows)}

    def delete(self, ids):
        for k in ids:
            self.rows.pop(k)

    def upsert(self, ids, documents, embeddings, metadatas):
        for k, d, v, m in zip(ids, documents, embeddings, metadatas):
            self.rows[k] = (d, v, m)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        items = list(self.rows.values())[:n_results]
        return {
            "documents": [[x[0] for x in items]],
            "metadatas": [[x[2] for x in items]],
            "distances": [[0.2 for _ in items]],
        }


def make_chunk(text, number=1):
    return {
        "source": "demo.md",
        "chunk_id": f"demo.md#chunk_{number:03d}",
        "text": text,
        "line_start": 1,
        "line_end": 3,
        "file_type": ".md",
    }


class TestDay5(unittest.TestCase):
    def setUp(self):
        self.db = FakeCollection()
        self.model = FakeModel()
        self.docs = [make_chunk("超时允许有限重试")]

    def test_first_sync(self):
        r = sync_index(self.db, self.docs, self.model)
        self.assertEqual((r["added"], r["total"]), (1, 1))

    def test_second_sync_no_reembedding(self):
        sync_index(self.db, self.docs, self.model)
        r = sync_index(self.db, self.docs, self.model)
        self.assertEqual((r["added"], r["unchanged"]), (0, 1))
        self.assertEqual(len(self.model.calls), 1)

    def test_edit_replaces_old(self):
        sync_index(self.db, self.docs, self.model)
        r = sync_index(self.db, [make_chunk("超时后检查幂等性")], self.model)
        self.assertEqual((r["added"], r["removed"], self.db.count()), (1, 1, 1))

    def test_delete_stale(self):
        sync_index(self.db, self.docs, self.model)
        r = sync_index(self.db, [], self.model)
        self.assertEqual((r["removed"], self.db.count()), (1, 0))

    def test_search_restores_source(self):
        sync_index(self.db, self.docs, self.model)
        hits = search_index("超时怎么办", collection=self.db, embedder=self.model)
        self.assertEqual(hits[0]["source"], "demo.md")
        self.assertEqual(hits[0]["score"], 0.8)

    def test_empty_search_no_model_call(self):
        self.assertEqual(
            search_index("   ", collection=self.db, embedder=self.model), []
        )
        self.assertEqual(len(self.model.calls), 0)

    def test_stable_key_and_fingerprint(self):
        self.assertEqual(chunk_key(self.docs[0]), chunk_key(self.docs[0]))
        self.assertNotEqual(fingerprint(self.docs), fingerprint([make_chunk("改了")]))


if __name__ == "__main__":
    unittest.main()
