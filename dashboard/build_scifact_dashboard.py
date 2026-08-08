import csv
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "scifact"
OUTPUT = (
    PROJECT
    / "dashboard"
    / "integrated"
    / "data"
    / "scifact_dashboard_data.json"
)


SYSTEMS = {
    "BM25": RESULTS / "bm25" / "rankings.jsonl",
    "E5": RESULTS / "e5" / "rankings.jsonl",
    "RRF": RESULTS / "rrf" / "rankings.jsonl",
    "Cross-encoder": (
        RESULTS
        / "cross-encoder"
        / "rankings.jsonl"
    ),
}


METRIC_FILES = {
    "BM25": RESULTS / "bm25" / "metrics.json",
    "E5": RESULTS / "e5" / "metrics.json",
    "RRF": RESULTS / "rrf" / "metrics.json",
    "Cross-encoder": (
        RESULTS
        / "cross-encoder"
        / "metrics.json"
    ),
}


def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}, "
                    f"line {line_number}: {exc}"
                ) from exc

    return records


def find_corpus_file():
    explicit_candidates = [
        PROJECT / "data/scifact/corpus.jsonl",
        PROJECT / "data/scifact/data/corpus.jsonl",
        PROJECT / "data/scifact/scifact/corpus.jsonl",
        (
            PROJECT
            / "experiments/scifact/"
            "scifact-master/data/corpus.jsonl"
        ),
        (
            PROJECT
            / "experiments/scifact/"
            "scifact-master/corpus.jsonl"
        ),
    ]

    for path in explicit_candidates:
        if path.exists():
            return path

    search_roots = [
        PROJECT / "data" / "scifact",
        PROJECT / "experiments" / "scifact",
    ]

    discovered = []

    for root in search_roots:
        if root.exists():
            discovered.extend(
                path
                for path in root.rglob("corpus.jsonl")
                if path.is_file()
            )

    if discovered:
        return sorted(discovered)[0]

    raise FileNotFoundError(
        "Could not find the SciFact corpus.jsonl file."
    )


def normalize_metric_key(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def find_metric(metrics, aliases):
    normalized = {
        normalize_metric_key(key): value
        for key, value in metrics.items()
    }

    for alias in aliases:
        key = normalize_metric_key(alias)

        if key in normalized:
            try:
                return float(normalized[key])
            except (TypeError, ValueError):
                return None

    return None


def load_corpus(path):
    corpus = {}

    for record in read_jsonl(path):
        doc_id = str(
            record.get(
                "doc_id",
                record.get(
                    "document_id",
                    record.get("id", ""),
                ),
            )
        )

        if not doc_id:
            continue

        abstract = record.get("abstract", "")

        if isinstance(abstract, list):
            abstract = " ".join(
                str(sentence)
                for sentence in abstract
            )

        corpus[doc_id] = {
            "doc_id": doc_id,
            "title": record.get(
                "title",
                f"SciFact document {doc_id}",
            ),
            "abstract": str(abstract),
        }

    return corpus


def ranking_score(entry, system_name):
    if system_name == "Cross-encoder":
        return entry.get(
            "cross_encoder_score",
            entry.get("score"),
        )

    return entry.get("score")


def ranking_lookup(record, system_name):
    result = {}

    for item in record.get("ranking", []):
        doc_id = str(item.get("doc_id", ""))

        if not doc_id:
            continue

        result[doc_id] = {
            "rank": item.get("rank"),
            "score": ranking_score(
                item,
                system_name,
            ),
        }

    return result


for system_name, path in SYSTEMS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {system_name} rankings: {path}"
        )


corpus_file = find_corpus_file()
corpus = load_corpus(corpus_file)


ranking_records = {
    system_name: {
        str(record["claim_id"]): record
        for record in read_jsonl(path)
    }
    for system_name, path in SYSTEMS.items()
}


common_claim_ids = set.intersection(
    *[
        set(records.keys())
        for records in ranking_records.values()
    ]
)


def claim_sort_key(claim_id):
    try:
        return (0, int(claim_id))
    except ValueError:
        return (1, claim_id)


cases = []


for claim_id in sorted(
    common_claim_ids,
    key=claim_sort_key,
):
    cross_record = ranking_records[
        "Cross-encoder"
    ][claim_id]

    claim_text = cross_record.get(
        "claim",
        ranking_records["BM25"][claim_id].get(
            "claim",
            "",
        ),
    )

    lookups = {
        system_name: ranking_lookup(
            records[claim_id],
            system_name,
        )
        for system_name, records
        in ranking_records.items()
    }

    top_documents = []

    for cross_item in cross_record.get(
        "ranking",
        [],
    )[:20]:
        doc_id = str(
            cross_item.get("doc_id", "")
        )

        document = corpus.get(
            doc_id,
            {
                "title": (
                    f"SciFact document {doc_id}"
                ),
                "abstract": "",
            },
        )

        row = {
            "cross_encoder_rank": (
                lookups["Cross-encoder"]
                .get(doc_id, {})
                .get("rank")
            ),
            "document_id": doc_id,
            "title": document.get(
                "title",
                f"SciFact document {doc_id}",
            ),
            "abstract": document.get(
                "abstract",
                "",
            ),
            "cross_encoder_score": (
                lookups["Cross-encoder"]
                .get(doc_id, {})
                .get("score")
            ),
            "bm25_rank": (
                lookups["BM25"]
                .get(doc_id, {})
                .get("rank")
            ),
            "bm25_score": (
                lookups["BM25"]
                .get(doc_id, {})
                .get("score")
            ),
            "e5_rank": (
                lookups["E5"]
                .get(doc_id, {})
                .get("rank")
            ),
            "e5_score": (
                lookups["E5"]
                .get(doc_id, {})
                .get("score")
            ),
            "rrf_rank": (
                lookups["RRF"]
                .get(doc_id, {})
                .get("rank")
            ),
            "rrf_score": (
                lookups["RRF"]
                .get(doc_id, {})
                .get("score")
            ),
        }

        top_documents.append(row)

    cases.append(
        {
            "claim_id": claim_id,
            "claim": claim_text,
            "candidate_count": cross_record.get(
                "candidate_count",
                len(
                    cross_record.get(
                        "ranking",
                        [],
                    )
                ),
            ),
            "top_documents": top_documents,
        }
    )


metrics = []

for system_name, path in METRIC_FILES.items():
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    metrics.append(
        {
            "system": system_name,
            "MRR": find_metric(
                raw,
                [
                    "mrr",
                    "mean_reciprocal_rank",
                ],
            ),
            "Recall@10": find_metric(
                raw,
                [
                    "recall@10",
                    "recall_at_10",
                    "recall10",
                ],
            ),
            "nDCG@10": find_metric(
                raw,
                [
                    "ndcg@10",
                    "ndcg_at_10",
                    "ndcg10",
                ],
            ),
            "MAP@1000": find_metric(
                raw,
                [
                    "map@1000",
                    "map_at_1000",
                    "map1000",
                    "map",
                ],
            ),
        }
    )


significance_file = (
    RESULTS
    / "significance"
    / "paired_bootstrap_comparisons.csv"
)

significance = []

if significance_file.exists():
    with significance_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        significance = list(
            csv.DictReader(file)
        )


evaluation_claim_file = (
    RESULTS
    / "cross-encoder"
    / "per_claim_metrics.jsonl"
)

evaluated_claim_count = (
    len(read_jsonl(evaluation_claim_file))
    if evaluation_claim_file.exists()
    else None
)


payload = {
    "metadata": {
        "dataset": "SciFact",
        "split": "development evidence claims",
        "ranked_claims": len(cases),
        "evaluated_claims": evaluated_claim_count,
        "corpus_documents": len(corpus),
        "corpus_file": str(corpus_file),
        "ranking_depth_displayed": 20,
    },
    "metrics": metrics,
    "significance": significance,
    "cases": cases,
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        payload,
        file,
        ensure_ascii=False,
        indent=2,
    )


print()
print("===== SCIFACT DASHBOARD EXPORT =====")
print("Corpus:", corpus_file)
print("Corpus documents:", len(corpus))
print("Claims:", len(cases))
print("Systems:", list(SYSTEMS.keys()))
print("Output:", OUTPUT)

if not cases:
    raise RuntimeError(
        "No common SciFact claims were exported."
    )

if not corpus:
    raise RuntimeError(
        "The SciFact corpus is empty."
    )

print("Validation: OK")
