"""
indexer.py

Addresses the P1 submission goal: "Retrieval across versions" -- codebases
change with every commit, so re-embedding the ENTIRE codebase from scratch
on every change would not scale. This module keeps a content-hash-keyed
cache of embeddings, so rebuilding the index for a new version only
re-embeds snippets that actually changed (added or edited); unchanged
snippets reuse their existing vector.

This is deliberately independent of MTEB/the encoder's exact model -- it
takes any object with an `.encode(list[str]) -> array`-like interface
(CodeRetrievalEncoder qualifies, and so does a fake one for testing), so it
can be unit-tested without network access or real model weights, same as
encoder.py.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IndexEntry:
    snippet_id: str
    content_hash: str
    vector: object  # embedding vector for this snippet's current content


@dataclass
class RebuildStats:
    total_snippets: int
    reused_from_cache: int
    newly_embedded: int
    removed: int


class CodeIndex:
    """A version-aware embedding index with incremental rebuilds.

    Usage:
        index = CodeIndex(encoder)
        stats = index.build_or_update({"snippet_1": "def f(): ...", ...})
        results = index.search("how is input validated?", top_k=5)
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self._entries: Dict[str, IndexEntry] = {}  # snippet_id -> IndexEntry

    def build_or_update(self, snippets: Dict[str, str]) -> RebuildStats:
        """Build (or incrementally update) the index for the given version
        of the codebase, represented as {snippet_id: code_text}.

        Snippets whose content hash matches what's already cached are
        reused as-is (no re-embedding); only new or changed snippets are
        sent to the encoder. Snippets no longer present in `snippets` are
        dropped from the index.
        """
        to_embed_ids: List[str] = []
        to_embed_texts: List[str] = []
        reused = 0

        for snippet_id, text in snippets.items():
            content_hash = _hash_text(text)
            existing = self._entries.get(snippet_id)
            if existing is not None and existing.content_hash == content_hash:
                reused += 1
                continue
            to_embed_ids.append(snippet_id)
            to_embed_texts.append(text)

        if to_embed_texts:
            vectors = self.encoder.encode(to_embed_texts)
            for snippet_id, text, vector in zip(to_embed_ids, to_embed_texts, vectors):
                self._entries[snippet_id] = IndexEntry(
                    snippet_id=snippet_id,
                    content_hash=_hash_text(text),
                    vector=vector,
                )

        # Drop snippets that no longer exist in this version.
        removed_ids = set(self._entries.keys()) - set(snippets.keys())
        for snippet_id in removed_ids:
            del self._entries[snippet_id]

        return RebuildStats(
            total_snippets=len(snippets),
            reused_from_cache=reused,
            newly_embedded=len(to_embed_texts),
            removed=len(removed_ids),
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return up to top_k (snippet_id, similarity_score) pairs, most
        relevant first, using cosine similarity between the query embedding
        and each indexed snippet's embedding."""
        if not self._entries:
            return []

        query_vector = self.encoder.encode([query])[0]
        scored = [
            (snippet_id, _cosine_similarity(query_vector, entry.vector))
            for snippet_id, entry in self._entries.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._entries)


def _cosine_similarity(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
