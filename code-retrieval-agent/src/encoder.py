"""
encoder.py

Implements the code-retrieval encoder required by the Samsung PRISM "Agentic
Code Intelligence" theme: given a natural-language query, rank code snippets
by relevance.

Architecture (per the official guideline's suggested improvement areas):
  1. Query pre-processing  (preprocess_query)   -- cleans/normalizes a query
  2. Snippet pre-processing (preprocess_snippet) -- cleans/normalizes code text
     before embedding (currently identity functions -- see Phase 2 notes below)
  3. Embedding              (CodeRetrievalEncoder.encode) -- turns text into
     vectors using a pretrained code-aware sentence-transformers model
  4. Ranking                -- handled by MTEB itself via cosine similarity
     over the embeddings this class returns

This class implements MTEB's real AbsEncoder interface (verified against the
installed mteb==2.21.0 API, not just the illustrative snippet in the
guideline): `encode(inputs, *, task_metadata, hf_split, hf_subset,
prompt_type=None, **kwargs)`, where `inputs` is a DataLoader yielding batches
that are dict-like objects with a "text" key (list[str]) for both queries and
corpus documents in a retrieval task.

The model is loaded lazily (only on first real encode() call) and can be
injected for testing, so the wiring logic here can be unit-tested without a
network connection or downloading any model weights -- see
tests/test_encoder.py.
"""
from mteb.models.abs_encoder import AbsEncoder

# The default model: a code-aware embedding model that runs on CPU.
# jina-embeddings-v2-base-code is a solid CPU-friendly baseline for code
# retrieval. If it's too slow/large in practice, swap for
# "flax-sentence-embeddings/st-codesearch-distilroberta-base" (smaller,
# faster, still code-aware) or "sentence-transformers/all-MiniLM-L6-v2"
# (general-purpose, fastest, weaker on code semantics).
DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"


def preprocess_query(text: str) -> str:
    """Normalize a natural-language query before embedding.

    Currently an identity function -- a clear, intentional seam for Phase 2
    improvements (e.g. query rewriting/expansion, stripping filler words,
    categorizing the query type) without touching the encoder's plumbing.
    """
    return text.strip()


def preprocess_snippet(text: str) -> str:
    """Normalize a code snippet before embedding.

    Currently an identity function -- a seam for Phase 2 improvements (e.g.
    stripping comments, normalizing whitespace/indentation, truncating very
    long snippets) without touching the encoder's plumbing.
    """
    return text.strip()


class CodeRetrievalEncoder(AbsEncoder):
    """Wraps a sentence-transformers code embedding model for MTEB.

    Parameters
    ----------
    model_name:
        A sentence-transformers-compatible model name/path. Defaults to a
        CPU-friendly code embedding model.
    model:
        An already-loaded model object exposing an `.encode(list[str],
        batch_size=...) -> array` method. Passing this in skips loading
        `model_name` entirely -- this is the seam that makes encode() unit
        testable without downloading real weights (see tests/test_encoder.py).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model=None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            # Imported here (not at module level) so this module can be
            # imported and unit-tested even in environments without
            # sentence-transformers' heavier dependencies fully configured.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type=None,
        **kwargs,
    ):
        """Embed every text in `inputs` and return an array of vectors.

        `inputs` is a DataLoader yielding batches; each batch is a dict-like
        object with a "text" key holding a list[str] -- this is true for
        both the query side and the corpus side of an MTEB retrieval task
        (see mteb.types._encoder_io.QueryInput / CorpusInput).

        Query-side batches are run through preprocess_query; everything else
        (corpus/document batches) is run through preprocess_snippet. MTEB
        signals which side we're on via `prompt_type` (PromptType.query vs
        PromptType.document/passage) when available; if that information
        isn't present we fall back to treating the batch as a corpus batch,
        which is the safer default for a code-retrieval task where the
        corpus vastly outnumbers queries.
        """
        is_query_side = prompt_type is not None and "query" in str(prompt_type).lower()
        preprocess = preprocess_query if is_query_side else preprocess_snippet

        texts: list[str] = []
        for batch in inputs:
            texts.extend(preprocess(t) for t in batch["text"])

        batch_size = kwargs.get("batch_size", 32)
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
