import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indexer import CodeIndex  # noqa: E402


class FakeEncoder:
    """Deterministic fake: each text's vector encodes its length, so
    similarity comparisons in tests are predictable, and we can count
    exactly which texts were sent for embedding."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, texts):
        self.encode_calls.append(list(texts))
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_first_build_embeds_everything():
    encoder = FakeEncoder()
    index = CodeIndex(encoder)

    stats = index.build_or_update({"a": "def add(x, y): return x + y", "b": "def sub(x, y): return x - y"})

    assert stats.total_snippets == 2
    assert stats.newly_embedded == 2
    assert stats.reused_from_cache == 0
    assert len(index) == 2


def test_rebuild_with_no_changes_reuses_everything():
    encoder = FakeEncoder()
    index = CodeIndex(encoder)
    snippets = {"a": "def add(x, y): return x + y"}

    index.build_or_update(snippets)
    encoder.encode_calls.clear()
    stats = index.build_or_update(snippets)

    assert stats.newly_embedded == 0
    assert stats.reused_from_cache == 1
    assert encoder.encode_calls == []  # confirms no re-embedding happened


def test_rebuild_only_reembeds_changed_snippet():
    encoder = FakeEncoder()
    index = CodeIndex(encoder)
    index.build_or_update({"a": "def add(x, y): return x + y", "b": "def sub(x, y): return x - y"})

    encoder.encode_calls.clear()
    stats = index.build_or_update({
        "a": "def add(x, y): return x + y",       # unchanged
        "b": "def subtract(x, y): return x - y",  # changed
    })

    assert stats.newly_embedded == 1
    assert stats.reused_from_cache == 1
    assert encoder.encode_calls == [["def subtract(x, y): return x - y"]]


def test_rebuild_removes_deleted_snippets():
    encoder = FakeEncoder()
    index = CodeIndex(encoder)
    index.build_or_update({"a": "text a", "b": "text b"})

    stats = index.build_or_update({"a": "text a"})  # "b" removed in this version

    assert stats.removed == 1
    assert len(index) == 1


class _DirectionalFakeEncoder:
    """Unlike FakeEncoder, this varies VECTOR DIRECTION (not just magnitude)
    with text length, so cosine similarity actually discriminates between
    snippets of different lengths -- needed because two vectors that only
    differ in magnitude along the same direction always have cosine
    similarity 1.0, which would make this test meaningless with the plain
    FakeEncoder above."""

    def encode(self, texts):
        return np.array([[float(len(t)), 100.0 - float(len(t))] for t in texts])


def test_search_ranks_by_similarity_and_respects_top_k():
    encoder = _DirectionalFakeEncoder()
    index = CodeIndex(encoder)
    index.build_or_update({"short": "ab", "medium": "abcdef", "long": "abcdefghijkl"})

    results = index.search("abcdef", top_k=2)  # query length 6, matches "medium" exactly

    assert len(results) == 2
    assert results[0][0] == "medium"  # closest length match ranks first


def test_search_on_empty_index_returns_empty_list():
    encoder = FakeEncoder()
    index = CodeIndex(encoder)
    assert index.search("anything") == []


if __name__ == "__main__":
    test_first_build_embeds_everything()
    test_rebuild_with_no_changes_reuses_everything()
    test_rebuild_only_reembeds_changed_snippet()
    test_rebuild_removes_deleted_snippets()
    test_search_ranks_by_similarity_and_respects_top_k()
    test_search_on_empty_index_returns_empty_list()
    print("All tests passed.")
