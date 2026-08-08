#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import evaluate
import joblib
import numpy as np
import textstat
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

CANDIDATES = ("beam4", "greedy", "concise")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.'’_-][A-Za-z0-9]+)*")
NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?%?(?!\w)",
    re.I,
)
NEGATIONS = {
    "no", "not", "none", "neither", "nor", "never", "without", "cannot",
    "can't", "didn't", "doesn't", "don't", "isn't", "wasn't", "weren't",
    "won't", "wouldn't", "shouldn't", "couldn't", "hasn't", "haven't", "hadn't",
}
UNCERTAINTY = {
    "may", "might", "could", "possibly", "possible", "probably", "likely",
    "unlikely", "suggest", "suggests", "suggested", "appears", "appear",
    "approximately", "about", "around", "estimated", "uncertain", "uncertainty",
    "potential", "potentially", "associated", "association",
    "not statistically significant", "statistically insignificant",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beam4", type=Path, required=True)
    p.add_argument("--greedy", type=Path, required=True)
    p.add_argument("--concise", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-model", default="intfloat/e5-base-v2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_jsonl(path):
    rows = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        eid = str(row.get("example_id", "")).strip()
        if not eid:
            raise ValueError(f"{path}:{i}: missing example_id")
        if eid in rows:
            raise ValueError(f"{path}: duplicate example_id {eid}")
        rows[eid] = row
    return rows


def pick(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"Missing one of fields: {keys}")


def source_of(row):
    return str(pick(row, ("source_text", "source", "input_text"))).strip()


def prediction_of(row):
    return str(pick(row, ("prediction", "output_text", "generated_text"))).strip()


def references_of(row):
    refs = row.get("references")
    if isinstance(refs, list) and refs:
        return [str(x).strip() for x in refs if str(x).strip()]
    return [str(pick(row, ("reference", "target_text"))).strip()]


def norm_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def toks(text):
    return [x.lower() for x in TOKEN_RE.findall(text)]


def normalize_number(x):
    x = x.lower().replace(",", "")
    suffix = "%" if x.endswith("%") else ""
    core = x[:-1] if suffix else x
    try:
        val = float(core)
        core = str(int(val)) if val.is_integer() else f"{val:.12g}"
    except ValueError:
        pass
    return core + suffix


def nums(text):
    return [normalize_number(m.group(0)) for m in NUMBER_RE.finditer(text)]


def multiset_prf(source_items, candidate_items):
    a, b = Counter(source_items), Counter(candidate_items)
    overlap = sum((a & b).values())
    precision = overlap / sum(b.values()) if b else 1.0
    recall = overlap / sum(a.values()) if a else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def uncertainty_terms(text):
    low = norm_space(text.lower())
    found = []
    for item in UNCERTAINTY:
        if (" " in item and item in low) or (
            " " not in item and re.search(rf"\b{re.escape(item)}\b", low)
        ):
            found.append(item)
    return found


def negation_terms(text):
    return [x for x in toks(text) if x in NEGATIONS]


def repeated_trigram_ratio(text):
    ts = toks(text)
    if len(ts) < 3:
        return 0.0
    grams = [tuple(ts[i:i + 3]) for i in range(len(ts) - 2)]
    counts = Counter(grams)
    return sum(v - 1 for v in counts.values() if v > 1) / len(grams)


def fkgl(text):
    try:
        value = float(textstat.flesch_kincaid_grade(text))
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def feature_dict(source, candidate):
    source, candidate = norm_space(source), norm_space(candidate)
    st, ct = toks(source), toks(candidate)
    ss, cs = set(st), set(ct)
    sim = SequenceMatcher(None, source.lower(), candidate.lower(), autojunk=False).ratio()
    np_, nr, nf = multiset_prf(nums(source), nums(candidate))
    gp, gr, gf = multiset_prf(negation_terms(source), negation_terms(candidate))
    up, ur, uf = multiset_prf(uncertainty_terms(source), uncertainty_terms(candidate))
    sf, cf = fkgl(source), fkgl(candidate)
    intersection, union = len(ss & cs), len(ss | cs)
    return {
        "char_similarity": sim,
        "token_jaccard": intersection / union if union else 1.0,
        "candidate_containment": sum(x in ss for x in ct) / max(len(ct), 1),
        "source_coverage": sum(x in cs for x in st) / max(len(st), 1),
        "source_words": float(len(st)),
        "candidate_words": float(len(ct)),
        "word_length_ratio": len(ct) / max(len(st), 1),
        "character_length_ratio": len(candidate) / max(len(source), 1),
        "source_fkgl": sf,
        "candidate_fkgl": cf,
        "readability_gain": sf - cf,
        "normalized_readability_gain": max(0.0, (sf - cf) / max(abs(sf), 1.0)),
        "number_precision": np_, "number_recall": nr, "number_f1": nf,
        "source_number_count": float(len(nums(source))),
        "candidate_number_count": float(len(nums(candidate))),
        "negation_precision": gp, "negation_recall": gr, "negation_f1": gf,
        "negation_count_difference": float(abs(len(negation_terms(source)) - len(negation_terms(candidate)))),
        "uncertainty_precision": up, "uncertainty_recall": ur, "uncertainty_f1": uf,
        "uncertainty_count_difference": float(abs(len(uncertainty_terms(source)) - len(uncertainty_terms(candidate)))),
        "repeated_trigram_ratio": repeated_trigram_ratio(candidate),
        "exact_copy": float(source == candidate),
        "near_copy_099": float(sim >= 0.99),
        "near_copy_095": float(sim >= 0.95),
        "near_copy_090": float(sim >= 0.90),
        "terminal_punctuation": float(bool(candidate) and candidate[-1] in ".!?"),
    }


def model(seed):
    return RandomForestRegressor(
        n_estimators=600, max_depth=6, min_samples_leaf=4,
        max_features=0.8, random_state=seed, n_jobs=-1,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems = {
        "beam4": load_jsonl(args.beam4),
        "greedy": load_jsonl(args.greedy),
        "concise": load_jsonl(args.concise),
    }
    example_ids = list(systems["beam4"])
    expected = set(example_ids)
    for name, rows in systems.items():
        if set(rows) != expected:
            raise ValueError(f"Example IDs differ for {name}")

    records, sources, predictions = [], [], []
    for eid in example_ids:
        anchor = systems["beam4"][eid]
        source, refs = source_of(anchor), references_of(anchor)
        for ctype in CANDIDATES:
            row = systems[ctype][eid]
            if norm_space(source_of(row)) != norm_space(source):
                raise ValueError(f"Source mismatch: {eid}/{ctype}")
            pred = prediction_of(row)
            rec = {
                "example_id": eid, "candidate_type": ctype,
                "source_text": source, "prediction": pred,
                "references": refs, "original_row": row,
            }
            rec.update(feature_dict(source, pred))
            records.append(rec)
            sources.append(source)
            predictions.append(pred)

    print(f"Loading semantic model: {args.semantic_model}")
    encoder = SentenceTransformer(args.semantic_model, device=args.device)
    src_emb = encoder.encode(
        [f"query: {x}" for x in sources], batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )
    pred_emb = encoder.encode(
        [f"query: {x}" for x in predictions], batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )
    for rec, score in zip(records, np.sum(src_emb * pred_emb, axis=1)):
        rec["semantic_cosine"] = float(score)

    print("Computing SARI targets...")
    sari = evaluate.load("sari")
    for i, rec in enumerate(records, 1):
        rec["target_sari"] = float(sari.compute(
            sources=[rec["source_text"]], predictions=[rec["prediction"]],
            references=[rec["references"]],
        )["sari"])
        if i % 50 == 0 or i == len(records):
            print(f"SARI targets: {i}/{len(records)}")

    for rec in records:
        for ctype in CANDIDATES:
            rec[f"candidate_is_{ctype}"] = float(rec["candidate_type"] == ctype)

    feature_names = [
        "char_similarity", "token_jaccard", "candidate_containment", "source_coverage",
        "source_words", "candidate_words", "word_length_ratio", "character_length_ratio",
        "source_fkgl", "candidate_fkgl", "readability_gain", "normalized_readability_gain",
        "number_precision", "number_recall", "number_f1", "source_number_count",
        "candidate_number_count", "negation_precision", "negation_recall", "negation_f1",
        "negation_count_difference", "uncertainty_precision", "uncertainty_recall",
        "uncertainty_f1", "uncertainty_count_difference", "repeated_trigram_ratio",
        "exact_copy", "near_copy_099", "near_copy_095", "near_copy_090",
        "terminal_punctuation", "semantic_cosine",
        "candidate_is_beam4", "candidate_is_greedy", "candidate_is_concise",
    ]
    X = np.asarray([[float(r[x]) for x in feature_names] for r in records])
    y = np.asarray([r["target_sari"] for r in records])
    groups = np.asarray([r["example_id"] for r in records])

    folds = min(args.folds, len(set(groups.tolist())))
    cv = GroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    predicted = np.full(len(records), np.nan)
    fold_ids = np.full(len(records), -1)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        reg = model(args.seed + fold)
        reg.fit(X[train_idx], y[train_idx])
        predicted[test_idx] = reg.predict(X[test_idx])
        fold_ids[test_idx] = fold
        print(f"Fold {fold}/{folds}: train={len(train_idx)}, held-out={len(test_idx)}")

    for rec, pred, fold in zip(records, predicted, fold_ids):
        rec["predicted_sari_cv"] = float(pred)
        rec["fold"] = int(fold)

    by_example = {}
    for rec in records:
        by_example.setdefault(rec["example_id"], []).append(rec)

    order = {name: i for i, name in enumerate(CANDIDATES)}
    selected_rows, selected_records = [], []
    selected_counts, oracle_counts = Counter(), Counter()
    selected_scores, oracle_scores, selected_sim = [], [], []

    for eid in example_ids:
        rows = by_example[eid]
        selected = max(rows, key=lambda r: (r["predicted_sari_cv"], -order[r["candidate_type"]]))
        oracle = max(rows, key=lambda r: (r["target_sari"], -order[r["candidate_type"]]))
        selected_records.append(selected)
        selected_counts[selected["candidate_type"]] += 1
        oracle_counts[oracle["candidate_type"]] += 1
        selected_scores.append(selected["target_sari"])
        oracle_scores.append(oracle["target_sari"])
        selected_sim.append(selected["char_similarity"])

        out = dict(selected["original_row"])
        out.update({
            "selected_from": selected["candidate_type"],
            "selector_fold": selected["fold"],
            "predicted_sari_cv": selected["predicted_sari_cv"],
            "actual_sari_for_analysis_only": selected["target_sari"],
        })
        selected_rows.append(out)

    candidate_means = {
        c: float(np.mean([r["target_sari"] for r in records if r["candidate_type"] == c]))
        for c in CANDIDATES
    }
    summary = {
        "examples": len(example_ids), "candidate_rows": len(records), "folds": folds,
        "candidate_mean_per_example_sari": candidate_means,
        "selected_counts": dict(selected_counts),
        "oracle_counts": dict(oracle_counts),
        "mean_cross_validated_selected_sari": float(np.mean(selected_scores)),
        "mean_oracle_sari": float(np.mean(oracle_scores)),
        "mean_oracle_regret": float(np.mean(oracle_scores) - np.mean(selected_scores)),
        "selected_copy_analysis": {
            "exact_source_copies": int(sum(r["exact_copy"] for r in selected_records)),
            "mean_source_similarity": float(np.mean(selected_sim)),
            "median_source_similarity": float(np.median(selected_sim)),
            "similarity_gte_0_99": int(sum(x >= 0.99 for x in selected_sim)),
            "similarity_gte_0_95": int(sum(x >= 0.95 for x in selected_sim)),
            "similarity_gte_0_90": int(sum(x >= 0.90 for x in selected_sim)),
        },
        "methodological_note": (
            "References create SARI targets only. All candidates of one example remain "
            "in the same fold; held-out selection uses reference-free features only."
        ),
    }

    with (args.output_dir / "selector_features.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            clean = {k: v for k, v in rec.items() if k != "original_row"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    with (args.output_dir / "selected_predictions_cv.jsonl").open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "selector_cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_model = model(args.seed)
    final_model.fit(X, y)
    joblib.dump(final_model, args.output_dir / "selector_model.joblib")
    importance = sorted(
        ({"feature": n, "importance": float(v)} for n, v in zip(feature_names, final_model.feature_importances_)),
        key=lambda x: x["importance"], reverse=True,
    )
    (args.output_dir / "feature_importance.json").write_text(
        json.dumps(importance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "selector_configuration.json").write_text(
        json.dumps({
            "semantic_model": args.semantic_model, "feature_names": feature_names,
            "candidate_order": CANDIDATES, "seed": args.seed, "folds": folds,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSELECTOR CROSS-VALIDATION COMPLETED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
