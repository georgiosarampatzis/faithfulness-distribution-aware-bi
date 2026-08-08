#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    return len(set(ranking[:k]) & relevant) / k


def ndcg_at_k(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
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


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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

    return f"passage: {title} {abstract_text}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E5 dense retrieval on SciFact."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="intfloat/e5-base-v2",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(args.data_dir / "corpus.jsonl")
    claims = load_jsonl(args.data_dir / "claims_dev.jsonl")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)

    model = SentenceTransformer(
        args.model,
        device="cuda",
    )

    doc_ids = [str(document["doc_id"]) for document in corpus]
    passages = [document_text(document) for document in corpus]

    cache_path = args.output_dir / "corpus_embeddings.npy"
    cache_ids_path = args.output_dir / "corpus_doc_ids.json"

    use_cache = (
        cache_path.exists()
        and cache_ids_path.exists()
        and not args.rebuild_cache
    )

    if use_cache:
        print("Loading cached corpus embeddings...")

        corpus_embeddings = np.load(cache_path)

        with cache_ids_path.open("r", encoding="utf-8") as handle:
            cached_doc_ids = json.load(handle)

        if cached_doc_ids != doc_ids:
            raise RuntimeError(
                "Cached document IDs do not match the current corpus. "
                "Run again with --rebuild-cache."
            )

        if corpus_embeddings.shape[0] != len(corpus):
            raise RuntimeError(
                "Cached embedding count does not match corpus size."
            )

    else:
        print(f"Encoding {len(passages)} corpus documents...")

        corpus_embeddings = model.encode(
            passages,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        np.save(cache_path, corpus_embeddings)

        with cache_ids_path.open("w", encoding="utf-8") as handle:
            json.dump(doc_ids, handle)

        print("Saved corpus embedding cache:", cache_path)

    query_texts = [
        f"query: {str(claim['claim']).strip()}"
        for claim in claims
    ]

    print(f"Encoding {len(query_texts)} claims...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Computing dense similarities...")

    similarities = np.matmul(
        query_embeddings,
        corpus_embeddings.T,
    )

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
        for claim_index, claim in enumerate(
            tqdm(claims, desc="Evaluating claims")
        ):
            claim_scores = similarities[claim_index]

            ranked_indices = np.argsort(
                -claim_scores,
                kind="stable",
            )[: args.top_k]

            ranking = [
                doc_ids[index]
                for index in ranked_indices
            ]

            ranking_record = {
                "claim_id": claim["id"],
                "claim": claim["claim"],
                "ranking": [
                    {
                        "rank": rank,
                        "doc_id": doc_ids[index],
                        "score": float(claim_scores[index]),
                    }
                    for rank, index in enumerate(
                        ranked_indices,
                        start=1,
                    )
                ],
            }

            rankings_file.write(
                json.dumps(
                    ranking_record,
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

            claim_metrics = {
                "claim_id": claim["id"],
                "gold_documents": sorted(relevant),
                "mrr": reciprocal_rank(ranking, relevant),
                "recall@1": recall_at_k(ranking, relevant, 1),
                "recall@3": recall_at_k(ranking, relevant, 3),
                "recall@5": recall_at_k(ranking, relevant, 5),
                "recall@10": recall_at_k(ranking, relevant, 10),
                "recall@20": recall_at_k(ranking, relevant, 20),
                "precision@10": precision_at_k(
                    ranking,
                    relevant,
                    10,
                ),
                "ndcg@10": ndcg_at_k(ranking, relevant, 10),
                "ndcg@20": ndcg_at_k(ranking, relevant, 20),
                "map@1000": average_precision(
                    ranking,
                    relevant,
                    1000,
                ),
            }

            metrics_file.write(
                json.dumps(
                    claim_metrics,
                    ensure_ascii=False,
                )
                + "\n"
            )

            for metric_name in metric_values:
                metric_values[metric_name].append(
                    float(claim_metrics[metric_name])
                )

    aggregate_metrics = {
        "system": "E5 dense retrieval",
        "model": args.model,
        "document_representation": "query/passsage prefixes; title + abstract",
        "similarity": "cosine via normalized dot product",
        "evaluation_split": "SciFact dev",
        "total_claims": len(claims),
        "evaluated_claims_with_gold_evidence": evaluated_claims,
        "corpus_documents": len(corpus),
        "embedding_dimension": int(corpus_embeddings.shape[1]),
        "top_k_saved": args.top_k,
    }

    for metric_name, values in metric_values.items():
        aggregate_metrics[metric_name] = mean(values)

    metrics_path = args.output_dir / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            aggregate_metrics,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nE5 evaluation completed.")
    print(json.dumps(aggregate_metrics, indent=2))
    print("\nRankings:", rankings_path)
    print("Per-claim metrics:", per_claim_path)
    print("Aggregate metrics:", metrics_path)


if __name__ == "__main__":
    main()
