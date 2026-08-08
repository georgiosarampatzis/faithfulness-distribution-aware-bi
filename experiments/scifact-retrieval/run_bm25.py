#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return rows


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranking[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = len(set(ranking[:k]) & relevant)
    return hits / k


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = 0.0
    for index, doc_id in enumerate(ranking[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(
    ranking: list[str],
    relevant: set[str],
    cutoff: int = 1000,
) -> float:
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(ranking[:cutoff], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BM25 abstract retrieval on SciFact."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing corpus.jsonl and claims_dev.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rankings and metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Maximum number of ranked documents to save.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.data_dir / "corpus.jsonl"
    claims_path = args.data_dir / "claims_dev.jsonl"

    corpus = load_jsonl(corpus_path)
    claims = load_jsonl(claims_path)

    doc_ids: list[str] = []
    tokenized_documents: list[list[str]] = []

    print(f"Loading and tokenizing {len(corpus)} abstracts...")

    for document in tqdm(corpus, desc="Corpus"):
        doc_id = str(document["doc_id"])
        title = str(document.get("title", ""))
        abstract = document.get("abstract", [])

        if isinstance(abstract, list):
            abstract_text = " ".join(str(sentence) for sentence in abstract)
        else:
            abstract_text = str(abstract)

        document_text = f"{title} {abstract_text}".strip()

        doc_ids.append(doc_id)
        tokenized_documents.append(tokenize(document_text))

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_documents)

    rankings_path = args.output_dir / "rankings.jsonl"
    per_claim_path = args.output_dir / "per_claim_metrics.jsonl"

    metric_values = {
        "mrr": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "recall@20": [],
        "precision@10": [],
        "ndcg@10": [],
        "ndcg@20": [],
        "map@1000": [],
    }

    evaluated_claims = 0

    with (
        rankings_path.open("w", encoding="utf-8") as rankings_file,
        per_claim_path.open("w", encoding="utf-8") as metrics_file,
    ):
        for claim in tqdm(claims, desc="Claims"):
            claim_id = claim["id"]
            claim_text = str(claim["claim"])

            scores = bm25.get_scores(tokenize(claim_text))
            ranked_indices = np.argsort(-scores, kind="stable")
            ranked_indices = ranked_indices[: args.top_k]

            ranking = [doc_ids[index] for index in ranked_indices]

            ranking_record = {
                "claim_id": claim_id,
                "claim": claim_text,
                "ranking": [
                    {
                        "rank": rank,
                        "doc_id": doc_ids[index],
                        "score": float(scores[index]),
                    }
                    for rank, index in enumerate(ranked_indices, start=1)
                ],
            }
            rankings_file.write(
                json.dumps(ranking_record, ensure_ascii=False) + "\n"
            )

            evidence = claim.get("evidence", {})
            relevant = {str(doc_id) for doc_id in evidence.keys()}

            # Claims without gold evidence are not used for retrieval scoring.
            if not relevant:
                continue

            evaluated_claims += 1

            claim_metrics = {
                "claim_id": claim_id,
                "gold_documents": sorted(relevant),
                "mrr": reciprocal_rank(ranking, relevant),
                "recall@1": recall_at_k(ranking, relevant, 1),
                "recall@3": recall_at_k(ranking, relevant, 3),
                "recall@5": recall_at_k(ranking, relevant, 5),
                "recall@10": recall_at_k(ranking, relevant, 10),
                "recall@20": recall_at_k(ranking, relevant, 20),
                "precision@10": precision_at_k(ranking, relevant, 10),
                "ndcg@10": ndcg_at_k(ranking, relevant, 10),
                "ndcg@20": ndcg_at_k(ranking, relevant, 20),
                "map@1000": average_precision(ranking, relevant, 1000),
            }

            metrics_file.write(
                json.dumps(claim_metrics, ensure_ascii=False) + "\n"
            )

            for metric_name in metric_values:
                metric_values[metric_name].append(
                    float(claim_metrics[metric_name])
                )

    aggregate_metrics = {
        "system": "BM25Okapi",
        "document_representation": "title + abstract",
        "evaluation_split": "SciFact dev",
        "total_claims": len(claims),
        "evaluated_claims_with_gold_evidence": evaluated_claims,
        "corpus_documents": len(corpus),
        "top_k_saved": args.top_k,
    }

    for metric_name, values in metric_values.items():
        aggregate_metrics[metric_name] = mean(values)

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate_metrics, handle, indent=2, ensure_ascii=False)

    print("\nBM25 evaluation completed.")
    print(json.dumps(aggregate_metrics, indent=2))
    print(f"\nRankings: {rankings_path}")
    print(f"Per-claim metrics: {per_claim_path}")
    print(f"Aggregate metrics: {metrics_path}")


if __name__ == "__main__":
    main()
