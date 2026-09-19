"""
demo.py

A small, self-contained demo using the exact example from the official
guideline (three snippets, one query) plus a couple more, so the demo video
can show real ranked retrieval results end to end -- not just a metrics JSON.

Requires internet access on first run (downloads the embedding model).

Usage:
    python -m src.demo
"""
from src.encoder import CodeRetrievalEncoder
from src.indexer import CodeIndex

# The exact example snippets from the official guideline, plus two extras
# for a slightly richer demo.
SNIPPETS = {
    "normalize": """function normalize(str) {
    const str2 = str.trim();
    return forward(str2);
}""",
    "check": """function check(s) {
    var pre = s.slice(0,6);
    return pre === 'en-US';
}""",
    "perf": """function perf(str) {
    if (act(A, str)) {
        return act(B, str)
    }
}""",
    "validate_email": """function validateEmail(email) {
    const cleaned = email.trim().toLowerCase();
    return /^[^@]+@[^@]+\\.[^@]+$/.test(cleaned);
}""",
    "retry_request": """async function retryRequest(fn, attempts) {
    for (let i = 0; i < attempts; i++) {
        try { return await fn(); } catch (e) { continue; }
    }
    throw new Error('all attempts failed');
}""",
}

QUERIES = [
    "How is the input preprocessed before going to the main function?",
    "How are failed network requests retried?",
]


def main():
    encoder = CodeRetrievalEncoder()
    index = CodeIndex(encoder)

    print("Building index over", len(SNIPPETS), "snippets...")
    stats = index.build_or_update(SNIPPETS)
    print(f"  embedded {stats.newly_embedded}, reused {stats.reused_from_cache}\n")

    for query in QUERIES:
        print(f"QUERY: {query}")
        results = index.search(query, top_k=3)
        for rank, (snippet_id, score) in enumerate(results, start=1):
            print(f"  #{rank}  {snippet_id}  (similarity: {score:.3f})")
        print()

    # Demonstrate P1: re-indexing after a code change only re-embeds what changed
    print("Simulating a code change to 'normalize'...")
    updated = dict(SNIPPETS)
    updated["normalize"] = """function normalize(str) {
    const str2 = str.trim().toLowerCase();
    return forward(str2);
}"""
    stats2 = index.build_or_update(updated)
    print(f"  Re-index stats: embedded {stats2.newly_embedded} (should be 1), "
          f"reused {stats2.reused_from_cache} (should be {len(SNIPPETS) - 1})")


if __name__ == "__main__":
    main()
