"""
Evaluate Egyptian NID OCR performance on a labeled dataset.

Usage:
    python scripts/evaluate_ocr.py --data-dir debug/eval_data --fields id_number name_ar address

Expected dataset format:
    - A directory (e.g. debug/eval_data) containing:
        - images/
            - <sample_id>.jpg|png|jpeg ...
        - labels.csv
    - labels.csv must be UTF-8 encoded with at least:
        - "filename" column: image filename (relative to images/ subfolder)
        - one column per field you want to evaluate (e.g. id_number,name_ar,address)

Example labels.csv:
    filename,id_number,name_ar,address
    sample_001.jpg,29901011234567,محمد كمال عبدالله,القاهرة
    sample_002.jpg,30101021234567,سارة أحمد علي,الجيزة
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.pipeline import IDExtractionPipeline


@dataclass
class SampleResult:
    filename: str
    field_results: Dict[str, bool]
    field_char_accuracy: Dict[str, float]
    processing_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Egyptian NID OCR pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to evaluation dataset directory (containing images/ and labels.csv).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["id_number"],
        help=(
            "List of field names to evaluate. Must match both labels.csv columns "
            "and pipeline output keys (e.g. id_number, name_ar, address)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of samples to evaluate (0 = all).",
    )
    return parser.parse_args()


def normalized_levenshtein(a: str, b: str) -> float:
    """Return character-level similarity in [0, 1] using Levenshtein distance."""

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        ca = a[i - 1]
        for j in range(1, len_b + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    dist = dp[len_a][len_b]
    max_len = max(len_a, len_b)
    return 1.0 - dist / max_len


def load_labels(
    data_dir: Path, fields: Sequence[str]
) -> List[Tuple[str, Dict[str, str]]]:
    """Load labels.csv and return list of (filename, field_dict)."""

    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")

    rows: List[Tuple[str, Dict[str, str]]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames:
            raise ValueError('labels.csv must contain a "filename" column')

        missing = [field for field in fields if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"labels.csv is missing columns for fields: {', '.join(missing)}"
            )

        for row in reader:
            filename = row["filename"].strip()
            if not filename:
                continue
            gt_fields = {field: (row.get(field) or "").strip() for field in fields}
            rows.append((filename, gt_fields))

    return rows


def evaluate_sample(
    pipeline: IDExtractionPipeline,
    images_dir: Path,
    filename: str,
    gt_fields: Dict[str, str],
) -> Optional[SampleResult]:
    """Run pipeline on one sample and compare to ground truth."""

    image_path = images_dir / filename
    if not image_path.exists():
        print(f"[WARN] Image not found for {filename}, skipping")
        return None

    with image_path.open("rb") as f:
        image_bytes = f.read()

    t0 = time.time()
    result = pipeline.process(image_bytes)
    t1 = time.time()

    # Prefer pipeline's own timing if available, fall back to wall time.
    processing_ms = int(result.get("processing_ms", (t1 - t0) * 1000))

    if "error" in result:
        print(f"[WARN] Pipeline error for {filename}: {result['error']}")
        field_results = {field: False for field in gt_fields}
        field_char_acc = {field: 0.0 for field in gt_fields}
        return SampleResult(
            filename=filename,
            field_results=field_results,
            field_char_accuracy=field_char_acc,
            processing_ms=processing_ms,
        )

    extracted: Dict[str, str] = result.get("extracted", {})
    field_results: Dict[str, bool] = {}
    field_char_acc: Dict[str, float] = {}

    for field, gt_value in gt_fields.items():
        pred = (extracted.get(field) or "").strip()
        gt_clean = gt_value.strip()

        if not gt_clean:
            # Skip metric for empty ground truth; mark as True with perfect char acc
            field_results[field] = True
            field_char_acc[field] = 1.0
            continue

        exact = pred == gt_clean
        sim = normalized_levenshtein(pred, gt_clean)
        field_results[field] = exact
        field_char_acc[field] = sim

    return SampleResult(
        filename=filename,
        field_results=field_results,
        field_char_accuracy=field_char_acc,
        processing_ms=processing_ms,
    )


def aggregate_results(
    results: Sequence[SampleResult], fields: Sequence[str]
) -> Dict[str, float]:
    """Compute KPIs from per-sample results."""

    metrics: Dict[str, float] = {}
    n = len(results)
    if n == 0:
        return metrics

    # Field-wise exact match accuracy and average char-level accuracy
    for field in fields:
        per_field_bools: List[bool] = []
        per_field_char: List[float] = []
        for r in results:
            if field in r.field_results:
                per_field_bools.append(r.field_results[field])
                per_field_char.append(r.field_char_accuracy[field])

        if per_field_bools:
            metrics[f"{field}.accuracy"] = sum(per_field_bools) / len(per_field_bools)
        if per_field_char:
            metrics[f"{field}.char_accuracy"] = sum(per_field_char) / len(
                per_field_char
            )

    # Document-level success: all evaluated fields correct
    doc_success = [
        all(r.field_results.get(field, False) for field in fields) for r in results
    ]
    metrics["document_success_rate"] = sum(doc_success) / len(doc_success)

    # Latency statistics
    latencies = [r.processing_ms for r in results]
    metrics["latency_ms.avg"] = sum(latencies) / len(latencies)
    metrics["latency_ms.median"] = statistics.median(latencies)
    metrics["latency_ms.p95"] = statistics.quantiles(latencies, n=20)[-1]
    metrics["latency_ms.max"] = max(latencies)

    return metrics


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    fields: List[str] = list(args.fields)

    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found at {images_dir}")

    print(f"[INFO] Loading labels from {data_dir}")
    label_rows = load_labels(data_dir, fields)
    if args.limit > 0:
        label_rows = label_rows[: args.limit]

    print(f"[INFO] Loaded {len(label_rows)} labeled samples")
    if not label_rows:
        print("[WARN] No labeled samples found, exiting")
        return

    pipeline = IDExtractionPipeline()

    results: List[SampleResult] = []
    for idx, (filename, gt_fields) in enumerate(label_rows, start=1):
        print(f"[INFO] [{idx}/{len(label_rows)}] Evaluating {filename} ...")
        sample_result = evaluate_sample(pipeline, images_dir, filename, gt_fields)
        if sample_result is not None:
            results.append(sample_result)

    print(f"[INFO] Completed {len(results)} evaluations")

    metrics = aggregate_results(results, fields)
    if not metrics:
        print("[WARN] No metrics computed")
        return

    print("\n=== OCR Evaluation Metrics ===")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if key.startswith("latency_ms."):
            print(f"{key:24s}: {value:8.1f}")
        else:
            print(f"{key:24s}: {value:8.4f}")


if __name__ == "__main__":
    main()

