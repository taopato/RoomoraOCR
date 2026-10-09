from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np
import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import main as legacy_service


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").upper())
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]", "", text)


def similarity(left: Any, right: Any) -> float:
    return SequenceMatcher(None, normalized_text(left), normalized_text(right)).ratio()


def parse_amount(value: Any) -> float | None:
    raw = str(value or "").replace("TL", "").replace(" ", "")
    if not raw:
        return None
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def date_key(value: Any) -> str:
    raw = str(value or "").strip()
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, pattern).date().isoformat()
        except ValueError:
            pass
    return ""


def item_pairs(expected: list[dict], actual: list[dict]) -> list[tuple[dict, dict, float]]:
    remaining = list(actual)
    pairs = []
    for expected_item in expected:
        if not remaining:
            break
        best = max(remaining, key=lambda item: similarity(expected_item.get("name"), item.get("name")))
        score = similarity(expected_item.get("name"), best.get("name"))
        remaining.remove(best)
        pairs.append((expected_item, best, score))
    return pairs


@dataclass
class ReceiptScore:
    receipt_id: str
    elapsed_seconds: float
    company_similarity: float
    date_correct: float
    total_correct: float
    item_name_f1: float
    item_amount_accuracy: float
    detected_items: int
    expected_items: int

    @property
    def quality(self) -> float:
        return (
            self.company_similarity * 0.10
            + self.date_correct * 0.10
            + self.total_correct * 0.30
            + self.item_name_f1 * 0.25
            + self.item_amount_accuracy * 0.25
        )


def score_receipt(receipt_id: str, expected: dict, actual: dict, elapsed: float) -> ReceiptScore:
    expected_items = expected.get("item_list") or []
    actual_items = actual.get("items") or []
    pairs = item_pairs(expected_items, actual_items)
    matched = sum(1 for _, _, score in pairs if score >= 0.75)
    precision = matched / max(len(actual_items), 1)
    recall = matched / max(len(expected_items), 1)
    item_name_f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    amount_matches = 0
    comparable = 0
    for expected_item, actual_item, name_score in pairs:
        if name_score < 0.75:
            continue
        expected_amount = parse_amount(expected_item.get("amount") or expected_item.get("price"))
        actual_amount = parse_amount(actual_item.get("line_total") or actual_item.get("lineTotal") or actual_item.get("price"))
        if expected_amount is None:
            continue
        comparable += 1
        if actual_amount is not None and abs(expected_amount - actual_amount) <= 0.02:
            amount_matches += 1

    expected_total = parse_amount(expected.get("total_amount"))
    actual_total = parse_amount(actual.get("total_amount") or actual.get("totalAmount"))
    total_correct = float(
        expected_total is not None and actual_total is not None and abs(expected_total - actual_total) <= 0.02
    )
    return ReceiptScore(
        receipt_id=receipt_id,
        elapsed_seconds=elapsed,
        company_similarity=similarity(expected.get("company"), actual.get("store_name") or actual.get("storeName")),
        date_correct=float(date_key(expected.get("date")) == date_key(actual.get("receipt_date") or actual.get("receiptDate"))),
        total_correct=total_correct,
        item_name_f1=item_name_f1,
        item_amount_accuracy=amount_matches / max(comparable, 1),
        detected_items=len(actual_items),
        expected_items=len(expected_items),
    )


def legacy_runner(image_bytes: bytes) -> dict:
    return legacy_service.run_ocr(image_bytes)


def rapidocr_runner(version: str, model_type: str, language: str) -> Callable[[bytes], dict]:
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    version_value = OCRVersion(version)
    model_value = ModelType(model_type)
    language_value = LangRec(language) if language in {item.value for item in LangRec} else language
    engine = RapidOCR(params={
        "Global.log_level": "warning",
        "Det.ocr_version": version_value,
        "Det.model_type": model_value,
        "Rec.ocr_version": version_value,
        "Rec.model_type": model_value,
        "Rec.lang_type": language_value,
    })

    def output_entries(output: Any) -> list[list[Any]]:
        boxes = output.boxes if output.boxes is not None else []
        texts = output.txts if output.txts is not None else []
        scores = output.scores if output.scores is not None else []
        return [
            [np.asarray(box).tolist(), text, float(score)]
            for box, text, score in zip(boxes, texts, scores)
        ]

    def mostly_vertical(entries: list[list[Any]]) -> bool:
        dimensions = [legacy_service.polygon_to_box(entry[0])[2:] for entry in entries]
        if len(dimensions) < 3:
            return False
        vertical = sum(1 for width, height in dimensions if height > width * 1.35)
        return vertical / len(dimensions) >= 0.60

    def run(image_bytes: bytes) -> dict:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"raw_text": "", "store_name": None, "receipt_date": None, "total_amount": None, "items": []}

        entries = output_entries(engine(image))
        if mostly_vertical(entries):
            rotated_results = [
                output_entries(engine(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))),
                output_entries(engine(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))),
            ]
            entries = max(rotated_results, key=legacy_service.ocr_result_score)

        lines = []
        for box, text, _ in entries:
            left, top, width, height = legacy_service.polygon_to_box(box)
            lines.append({"text": legacy_service.normalize_text(text), "left": left, "top": top, "width": width, "height": height})
        lines.sort(key=lambda item: (item["top"], item["left"]))
        merged = legacy_service.split_mixed_lines(legacy_service.merge_lines(lines))
        return {
            "raw_text": "\n".join(line["text"] for line in merged),
            "store_name": next((line["text"] for line in merged[:8] if legacy_service.store_line_score(line["text"]) >= 8), None),
            "receipt_date": legacy_service.try_extract_date(merged),
            "total_amount": legacy_service.try_extract_total(merged),
            "items": legacy_service.extract_items(merged),
        }

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate free OCR engines on the Roomora receipt benchmark.")
    parser.add_argument("--dataset", type=Path, default=Path(".benchmark-data"))
    parser.add_argument("--engine", choices=("production", "v5-latin", "v6-small", "v6-medium"), default="production")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--receipt-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runners = {
        "production": legacy_runner,
        "v5-latin": lambda: rapidocr_runner("PP-OCRv5", "mobile", "latin"),
        "v6-small": lambda: rapidocr_runner("PP-OCRv6", "small", "tr"),
        "v6-medium": lambda: rapidocr_runner("PP-OCRv6", "medium", "tr"),
    }
    runner = runners[args.engine] if args.engine == "production" else runners[args.engine]()
    root = args.dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if args.receipt_id:
        manifest = [record for record in manifest if record["id"] == args.receipt_id]
        if not manifest:
            raise SystemExit(f"Receipt not found: {args.receipt_id}")
    if args.limit:
        manifest = manifest[:args.limit]

    scores = []
    predictions = []
    for index, record in enumerate(manifest, start=1):
        image_bytes = (root / record["image"]).read_bytes()
        expected = json.loads((root / record["annotation"]).read_text(encoding="utf-8"))
        started = time.perf_counter()
        actual = runner(image_bytes)
        elapsed = time.perf_counter() - started
        score = score_receipt(record["id"], expected, actual, elapsed)
        scores.append(score)
        predictions.append({"id": record["id"], "expected": expected, "actual": actual, "score": asdict(score) | {"quality": score.quality}})
        print(f"{args.engine} {index}/{len(manifest)} quality={score.quality:.3f} time={elapsed:.2f}s")

    summary = {
        "engine": args.engine,
        "receipts": len(scores),
        "quality": statistics.fmean(score.quality for score in scores),
        "company_similarity": statistics.fmean(score.company_similarity for score in scores),
        "date_accuracy": statistics.fmean(score.date_correct for score in scores),
        "total_accuracy": statistics.fmean(score.total_correct for score in scores),
        "item_name_f1": statistics.fmean(score.item_name_f1 for score in scores),
        "item_amount_accuracy": statistics.fmean(score.item_amount_accuracy for score in scores),
        "average_seconds": statistics.fmean(score.elapsed_seconds for score in scores),
        "median_seconds": statistics.median(score.elapsed_seconds for score in scores),
    }
    payload = {"summary": summary, "predictions": predictions}
    output = args.output or root / f"results-{args.engine}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
