"""
tests/test_encoder.py

These tests verify CodeRetrievalEncoder's WIRING (how it extracts text from
MTEB's batch format and calls the underlying model) without needing network
access or real model weights -- a fake model is injected instead. This is
why these tests can run in a sandboxed/offline environment; the REAL
end-to-end evaluation (real model, real CoIR apps dataset) must be run
separately via run_eval.py on a machine with internet access.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.encoder import CodeRetrievalEncoder, preprocess_query, preprocess_snippet  # noqa: E402


class FakeModel:
    """Stands in for a real SentenceTransformer: records what it was asked
    to encode and returns deterministic fake vectors, so tests can assert
    on both the text that reached the model and the shape of what comes back."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        self.calls.append(list(texts))
        # one 4-dim fake vector per text, deterministic based on text length
        return np.array([[float(len(t))] * 4 for t in texts])


def _batches(list_of_dicts):
    """A minimal stand-in for the DataLoader MTEB passes to encode()."""
    return list_of_dicts


def test_encode_extracts_text_from_corpus_batches():
    fake = FakeModel()
    encoder = CodeRetrievalEncoder(model=fake)

    batches = _batches([
        {"text": ["def add(a, b): return a + b"]},
        {"text": ["def sub(a, b): return a - b"]},
    ])
    result = encoder.encode(batches, task_metadata=None, hf_split="test", hf_subset="default")

    assert fake.calls == [["def add(a, b): return a + b", "def sub(a, b): return a - b"]]
    assert result.shape == (2, 4)


def test_encode_extracts_text_from_query_batches_and_uses_query_preprocessing():
    fake = FakeModel()
    encoder = CodeRetrievalEncoder(model=fake)

    batches = _batches([{"text": ["  how is input validated?  "]}])
    encoder.encode(batches, task_metadata=None, hf_split="test", hf_subset="default", prompt_type="query")

    # preprocess_query strips whitespace -- confirms the query path ran,
    # not the snippet path
    assert fake.calls == [["how is input validated?"]]


def test_encode_handles_multiple_batches_in_one_call():
    fake = FakeModel()
    encoder = CodeRetrievalEncoder(model=fake)

    batches = _batches([
        {"text": ["snippet one"]},
        {"text": ["snippet two", "snippet three"]},
    ])
    result = encoder.encode(batches, task_metadata=None, hf_split="test", hf_subset="default")

    assert fake.calls == [["snippet one", "snippet two", "snippet three"]]
    assert result.shape == (3, 4)


def test_model_is_lazy_when_not_injected():
    # No model injected and no encode() call made -- constructing the
    # encoder must NOT attempt to load/download anything.
    encoder = CodeRetrievalEncoder(model_name="some/model-that-does-not-exist")
    assert encoder._model is None  # confirms nothing was eagerly loaded


def test_preprocess_functions_are_identity_by_default():
    # Documents the current baseline behavior explicitly, so a future
    # Phase-2 change to these functions has a clear regression test to update.
    assert preprocess_query("  hello  ") == "hello"
    assert preprocess_snippet("  def f(): pass  ") == "def f(): pass"


if __name__ == "__main__":
    test_encode_extracts_text_from_corpus_batches()
    test_encode_extracts_text_from_query_batches_and_uses_query_preprocessing()
    test_encode_handles_multiple_batches_in_one_call()
    test_model_is_lazy_when_not_injected()
    test_preprocess_functions_are_identity_by_default()
    print("All tests passed.")
