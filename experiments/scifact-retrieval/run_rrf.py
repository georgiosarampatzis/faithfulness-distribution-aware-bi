#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


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


def load_rankings(path: Path) -> dict[str, dict]:
    return {
        str(row["claim_id"]): row
        for row in load_jsonl(path)
    }


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / k


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, doc_id in enumerate(ranking[:k])
        if doc_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse BM25 and E5 SciFact rankings using RRF."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bm25-rankings", type=Path, required=True)
    parser.add_argument("--e5-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    claims = load_jsonl(args.data_dir / "claims_dev.jsonl")
    bm25 = load_rankings(args.bm25_rankings)
    e5 = load_rankings(args.e5_rankings)

    metric_names = [
        "mrr",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "recall@20",
        "precision@10",
        "ndcg@10",
        "ndcg@20",
        "map@1000",
    ]

    metric_values = {
        name: []
        for name in metric_names
    }

    rankings_path = args.output_dir / "rankings.jsonl"
    per_claim_path = args.output_dir / "per_claim_metrics.jsonl"

    evaluated_claims = 0

    with (
        rankings_path.open("w", encoding="utf-8") as rankings_file,
        per_claim_path.open("w", encoding="utf-8") as metrics_file,
    ):
        for claim in tqdm(claims, desc="RRF fusion"):
            claim_key = str(claim["id"])

            if claim_key not in bm25:
                raise KeyError(f"Claim {claim_key} missing from BM25 rankings.")

            if claim_key not in e5:
                raise KeyError(f"Claim {claim_key} missing from E5 rankings.")

            scores: dict[str, float] = {}
            source_ranks: dict[str, dict[str, int]] = {}

            for system_name, system_rows in [
                ("bm25", bm25[claim_key]["ranking"]),
                ("e5", e5[claim_key]["ranking"]),
            ]:
                for item in system_rows[: args.top_k]:
                    doc_id = str(item["doc_id"])
                    rank = int(item["rank"])

                    scores[doc_id] = scores.get(doc_id, 0.0) + (
                        1.0 / (args.rrf_k + rank)
                    )

                    source_ranks.setdefault(doc_id, {})[system_name] = rank

            ranked_doc_ids = sorted(
                scores,
                key=lambda doc_id: (
                    -scores[doc_id],
                    min(source_ranks[doc_id].values()),
                    doc_id,
                ),
            )[: args.top_k]

            rankings_file.write(
                json.dumps(
                    {
                        "claim_id": claim["id"],
                        "claim": claim["claim"],
                        "ranking": [
                            {
                                "rank": rank,
                                "doc_id": doc_id,
                                "score": scores[doc_id],
                                "source_ranks": source_ranks[doc_id],
                            }
                            for rank, doc_id in enumerate(
                                ranked_doc_ids,
                                start=1,
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            relevant = {
                str(doc_id)
                for doc_id in claim.get("evidence", {}).keys()
            }

            if not relevant:
                continue

            evaluated_claims += 1

            values = {
                "claim_id": claim["id"],
                "gold_documents": sorted(relevant),
                "mrr": reciprocal_rank(ranked_doc_ids, relevant),
                "recall@1": recall_at_k(ranked_doc_ids, relevant, 1),
                "recall@3": recall_at_k(ranked_doc_ids, relevant, 3),
                "recall@5": recall_at_k(ranked_doc_ids, relevant, 5),
                "recall@10": recall_at_k(ranked_doc_ids, relevant, 10),
                "recall@20": recall_at_k(ranked_doc_ids, relevant, 20),
                "precision@10": precision_at_k(
                    ranked_doc_ids,
                    relevant,
                    10,
                ),
                "ndcg@10": ndcg_at_k(ranked_doc_ids, relevant, 10),
                "ndcg@20": ndcg_at_k(ranked_doc_ids, relevant, 20),
                "map@1000": average_precision(
                    ranked_doc_ids,
                    relevant,
                    1000,
                ),
            }

            metrics_file.write(
                json.dumps(values, ensure_ascii=False) + "\n"
            )

            for metric_name in metric_names:
                metric_values[metric_name].append(
                    float(values[metric_name])
                )

    aggregate = {
        "system": "Reciprocal Rank Fusion",
        "components": ["BM25Okapi", "intfloat/e5-base-v2"],
        "rrf_k": args.rrf_k,
        "evaluation_split": "SciFact dev",
        "total_claims": len(claims),
        "evaluated_claims_with_gold_evidence": evaluated_claims,
        "top_k_saved": args.top_k,
    }

    for metric_name, values in metric_values.items():
        aggregate[metric_name] = float(np.mean(values))

    metrics_path = args.output_dir / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            aggregate,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nRRF evaluation completed.")
    print(json.dumps(aggregate, indent=2))
    print("\nRankings:", rankings_path)
    print("Per-claim metrics:", per_claim_path)
    print("Aggregate metrics:", metrics_path)


if __name__ == "__main__":
    main()
