"""
run_eval.py

Runs the official evaluation required by the Samsung PRISM "Agentic Code
Intelligence" submission: MTEB's "AppsRetrieval" task (backed by the
CoIR-Retrieval/apps dataset), scored by NDCG@10 and MRR.

USAGE (requires internet access -- downloads the model and dataset from
Hugging Face on first run, which can take a while and use several GB of
disk depending on the chosen model):

    python -m src.run_eval

Produces: appsretrieval_results.json in the project root -- this is the
file to attach to the GitHub release (tag: PRISM_GENAI_HACKATHON_Y2026) per
the official submission instructions.
"""
import json
import sys

from src.encoder import CodeRetrievalEncoder


def main(output_path: str = "appsretrieval_results.json") -> dict:
    import mteb

    model = CodeRetrievalEncoder()
    task = mteb.get_task("AppsRetrieval")

    print(f"Running MTEB task: {task.metadata.name}")
    print(f"Dataset: {task.metadata.dataset}")
    print("This will download the dataset and model on first run -- this may take a while.\n")

    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})
    task_result = list(result.task_results)[0]
    result_dict = task_result.to_dict()

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nSaved results to {output_path}")
    _print_headline_scores(result_dict)
    return result_dict


def _print_headline_scores(result_dict: dict) -> None:
    """Best-effort extraction of NDCG@10 / MRR for a human-readable summary
    printed to the console -- the full detail lives in the JSON itself,
    whose exact structure is produced by MTEB and shouldn't be hand-parsed
    for the actual submission."""
    try:
        scores = result_dict.get("scores", {})
        for split, split_scores in scores.items():
            for entry in split_scores:
                ndcg = entry.get("ndcg_at_10")
                mrr = entry.get("mrr")
                print(f"[{split}] NDCG@10 = {ndcg}, MRR = {mrr}")
    except Exception:
        print("(Could not pretty-print headline scores -- see the JSON file for full results.)")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "appsretrieval_results.json"
    main(output)
