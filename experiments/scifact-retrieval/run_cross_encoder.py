#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from run_rrf import (
    average_precision,
    load_jsonl,
    load_rankings,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def document_text(document: dict) -> str:
    title = str(document.get("title", "")).strip()
    abstract = document.get("abstract", [])

    if isinstance(abstract, list):
        abstract_text = " ".join(
            str(sentence).strip()
            for sentence in abstract
            if str(sentence).strip()
        )
    else:
        abstract_text = str(abstract).strip()

    return f"{title} {abstract_text}".strip()


def ranking_doc_ids(record: dict, depth: int | None = None) -> list[str]:
    rows = record["ranking"]

    if depth is not None:
        rows = rows[:depth]

    return [str(item["doc_id"]) for item in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-encoder reranking for SciFact retrieval."
    )

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bm25-rankings", type=Path, required=True)
    parser.add_argument("--e5-rankings", type=Path, required=True)
    parser.add_argument("--rrf-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=100,
        help="Take this many candidates from both BM25 and E5.",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=1000)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    corpus_rows = load_jsonl(args.data_dir / "corpus.jsonl")
    claims = load_jsonl(args.data_dir / "claims_dev.jsonl")

    corpus = {
        str(document["doc_id"]): document_text(document)
        for document in corpus_rows
    }

    bm25 = load_rankings(args.bm25_rankings)
    e5 = load_rankings(args.e5_rankings)
    rrf = load_rankings(args.rrf_rankings)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Cross-encoder:", args.model)
    print("Candidate depth per retriever:", args.candidate_depth)

    model = CrossEncoder(
        args.model,
        device="cuda",
        max_length=args.max_length,
    )

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
        "candidate_recall",
    ]

    metric_values = {
        metric_name: []
        for metric_name in metric_names
    }

    rankings_path = args.output_dir / "rankings.jsonl"
    per_claim_path = args.output_dir / "per_claim_metrics.jsonl"
    evaluated_claims = 0
    candidate_sizes = []

    with (
        rankings_path.open("w", encoding="utf-8") as rankings_file,
        per_claim_path.open("w", encoding="utf-8") as metrics_file,
    ):
        for claim in tqdm(claims, desc="Cross-encoder reranking"):
            claim_key = str(claim["id"])
            claim_text = str(claim["claim"]).strip()

            for system_name, system in [
                ("BM25", bm25),
                ("E5", e5),
                ("RRF", rrf),
            ]:
                if claim_key not in system:
                    raise KeyError(
                        f"Claim {claim_key} is missing from {system_name}."
                    )

            bm25_candidates = ranking_doc_ids(
                bm25[claim_key],
                args.candidate_depth,
            )

            e5_candidates = ranking_doc_ids(
                e5[claim_key],
                args.candidate_depth,
            )

            candidate_set = set(bm25_candidates) | set(e5_candidates)

            # Use RRF order only as a deterministic initial ordering.
            rrf_order = ranking_doc_ids(
                rrf[claim_key],
                args.top_k,
            )

            candidates = [
                doc_id
                for doc_id in rrf_order
                if doc_id in candidate_set
            ]

            # Defensive fallback in case an item is absent from RRF output.
            seen_candidates = set(candidates)
            candidates.extend(
                doc_id
                for doc_id in sorted(candidate_set)
                if doc_id not in seen_candidates
            )

            missing_from_corpus = [
                doc_id
                for doc_id in candidates
                if doc_id not in corpus
            ]

            if missing_from_corpus:
                raise KeyError(
                    f"Missing corpus documents for claim {claim_key}: "
                    f"{missing_from_corpus[:10]}"
                )

            candidate_sizes.append(len(candidates))

            pairs = [
                (claim_text, corpus[doc_id])
                for doc_id in candidates
            ]

            scores = model.predict(
                pairs,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            scores = np.asarray(scores).reshape(-1)

            reranked_pairs = sorted(
                zip(candidates, scores),
                key=lambda item: (-float(item[1]), item[0]),
            )

            reranked_candidates = [
                doc_id
                for doc_id, _ in reranked_pairs
            ]

            reranker_scores = {
                doc_id: float(score)
                for doc_id, score in reranked_pairs
            }

            candidate_set_final = set(reranked_candidates)

            # Append the rest of the RRF ranking after reranked candidates.
            final_ranking = reranked_candidates + [
                doc_id
                for doc_id in rrf_order
                if doc_id not in candidate_set_final
            ]

            final_ranking = final_ranking[: args.top_k]

            rankings_file.write(
                json.dumps(
                    {
                        "claim_id": claim["id"],
                        "claim": claim_text,
                        "candidate_count": len(candidates),
                        "ranking": [
                            {
                                "rank": rank,
                                "doc_id": doc_id,
                                "cross_encoder_score": (
                                    reranker_scores.get(doc_id)
                                ),
                                "reranked_candidate": (
                                    doc_id in candidate_set_final
                                ),
                            }
                            for rank, doc_id in enumerate(
                                final_ranking,
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

            candidate_recall = (
                len(candidate_set_final & relevant) / len(relevant)
            )

            claim_metrics = {
                "claim_id": claim["id"],
                "gold_documents": sorted(relevant),
                "candidate_count": len(candidates),
                "candidate_recall": candidate_recall,
                "mrr": reciprocal_rank(final_ranking, relevant),
                "recall@1": recall_at_k(final_ranking, relevant, 1),
                "recall@3": recall_at_k(final_ranking, relevant, 3),
                "recall@5": recall_at_k(final_ranking, relevant, 5),
                "recall@10": recall_at_k(final_ranking, relevant, 10),
                "recall@20": recall_at_k(final_ranking, relevant, 20),
                "precision@10": precision_at_k(
                    final_ranking,
                    relevant,
                    10,
                ),
                "ndcg@10": ndcg_at_k(final_ranking, relevant, 10),
                "ndcg@20": ndcg_at_k(final_ranking, relevant, 20),
                "map@1000": average_precision(
                    final_ranking,
                    relevant,
                    1000,
                ),
            }

            metrics_file.write(
                json.dumps(claim_metrics, ensure_ascii=False) + "\n"
            )

            for metric_name in metric_names:
                metric_values[metric_name].append(
                    float(claim_metrics[metric_name])
                )

    aggregate = {
        "system": "Cross-encoder reranking",
        "model": args.model,
        "first_stage": "union of BM25 and E5 candidates",
        "candidate_depth_per_retriever": args.candidate_depth,
        "evaluation_split": "SciFact dev",
        "total_claims": len(claims),
        "evaluated_claims_with_gold_evidence": evaluated_claims,
        "corpus_documents": len(corpus),
        "average_candidate_count": float(np.mean(candidate_sizes)),
        "maximum_candidate_count": int(max(candidate_sizes)),
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

    print("\nCross-encoder evaluation completed.")
    print(json.dumps(aggregate, indent=2))
    print("\nRankings:", rankings_path)
    print("Per-claim metrics:", per_claim_path)
    print("Aggregate metrics:", metrics_path)


if __name__ == "__main__":
    main()
