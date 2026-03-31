from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Receipt OCR Service", version="2.1.0")
ocr_engine: RapidOCR | None = None

IGNORE_KEYWORDS = [
    "TOPLAM",
    "ARA TOPLAM",
    "KDV",
    "NAKIT",
    "PARA USTU",
    "FIS NO",
    "Z NO",
    "TARIH",
    "SAAT",
    "KASIYER",
    "ALINAN PARA",
    "TCKN",
    "VKN",
    "VERGI",
    "HESAP ADI",
    "HESAP KODU",
    "BAKIYE",
    "NÄ°HAI",
    "NIHAI",
    "ODENECEK",
    "TUTAR",
    "VERESIYE",
    "KDV FISI",
]

DECIMAL_AMOUNT_PATTERN = re.compile(r"\d{1,3}(?:[ .]\d{3})*(?:[.,]\d{2})")
LETTER_PATTERN = re.compile(r"[A-ZÇĞİÖŞÜa-zçğıöşü]")
DATE_PATTERN = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{2,4})")
MEASUREMENT_UNITS = ("L", "LT", "ML", "GR", "G", "KG", "CL")
STORE_BLACKLIST = {
    "TARIH",
    "SAAT",
    "FIS",
    "FIŞ",
    "NO",
    "KREDI",
    "KREDİ",
    "KARTI",
    "HALKBANK",
    "ODEME",
    "ÖDEME",
    "POS",
    "TOPLAM",
    "KDV",
}
MONTH_HINTS = {
    "OCAK",
    "SUBAT",
    "ŞUBAT",
    "MART",
    "NISAN",
    "NİSAN",
    "MAYIS",
    "HAZIRAN",
    "TEMMUZ",
    "AGUSTOS",
    "AĞUSTOS",
    "EYLUL",
    "EYLÜL",
    "EKIM",
    "EKİM",
    "KASIM",
    "ARALIK",
}


def get_ocr() -> RapidOCR:
    global ocr_engine
    if ocr_engine is None:
        ocr_engine = RapidOCR()
    return ocr_engine


def normalize_text(value: str) -> str:
    cleaned = (
        (value or "")
        .replace("ï¿¥", " ")
        .replace("¥", " ")
        .replace("â‚º", " ")
    )
    return re.sub(r"\s+", " ", cleaned.strip())


def parse_amount(raw: str) -> float | None:
    value = normalize_text(raw).replace("TL", "").replace(" ", "")
    if not value:
        return None

    if value.count(",") > 1 or value.count(".") > 1 or ("," in value and "." in value):
        value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def amount_tokens(text: str) -> list[re.Match[str]]:
    return list(DECIMAL_AMOUNT_PATTERN.finditer(normalize_text(text)))


def amount_from_text(text: str) -> float | None:
    matches = amount_tokens(text)
    if not matches:
        return None
    return parse_amount(matches[-1].group(0))


def amount_only(text: str) -> float | None:
    normalized = normalize_text(text)
    cleaned = normalized.replace("TL", "").strip()
    cleaned = re.sub(r"^[*xX+\-]+", "", cleaned).strip()
    if re.fullmatch(DECIMAL_AMOUNT_PATTERN.pattern, cleaned):
        return parse_amount(cleaned)
    return None


def has_letters(text: str) -> bool:
    return bool(LETTER_PATTERN.search(normalize_text(text)))


def is_code_or_meta(text: str) -> bool:
    upper = normalize_text(text).upper()
    if not upper:
        return True
    if any(keyword in upper for keyword in IGNORE_KEYWORDS):
        return True
    if any(keyword in upper for keyword in ["KART", "POS", "TEKPOS", "BANK", "VISA", "MASTER", "MASTERCARD"]):
        return True
    if re.fullmatch(r"\d{6,}", upper):
        return True
    if re.match(r"^\d+\s*\(", upper):
        return True
    if "ADET X" in upper:
        return True
    return False


def is_product_name(text: str) -> bool:
    upper = normalize_text(text).upper()
    if not has_letters(upper):
        return False
    if re.search(r"\bADE[TL]X?\b|\bADET\b", upper):
        return False
    return not is_code_or_meta(upper)


def normalize_product_candidate(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"^[\W_]+|[\W_]+$", "", value)
    value = re.sub(r"^\d{6,}\s*", "", value)
    value = re.sub(r"\(\s*\d+\s*ADET\s*X\s*\d+[.,]\d{2}\s*\)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"%\s*\d+[.,]?\d*", "", value)
    value = re.sub(r"\bADET\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bX\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[*xX]?\s*\d{1,4}(?:[.,]\d{2})?$", "", value)
    value = re.sub(r"\b\d{1,2}[.,]\d{1,2}\s*(?:L|LT|ML|GR|G|KG|CL)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+\s*(?:L|LT|ML|GR|G|KG|CL)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value)
    return normalize_text(value)


def product_candidate_score(candidate: str) -> int:
    value = normalize_product_candidate(candidate)
    if not value or is_code_or_meta(value) or not has_letters(value):
        return -100

    letters = len(re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]", value))
    digits = len(re.findall(r"\d", value))
    words = len(value.split())
    score = min(letters, 24) + min(words * 3, 18) - min(digits * 2, 12)

    upper = value.upper()
    if re.search(r"[*%]", upper):
        score -= 8
    if any(keyword in upper for keyword in STORE_BLACKLIST):
        score -= 20
    if len(value) < 3:
        score -= 20

    return score


def store_line_score(text: str) -> int:
    value = normalize_text(text)
    upper = value.upper()
    if not value or is_code_or_meta(value) or amount_only(value):
        return -100
    if DATE_PATTERN.search(value):
        return -100
    if ":" in value:
        return -60
    if sum(ch.isdigit() for ch in value) >= max(2, len(value) // 4):
        return -50
    if any(token in upper for token in STORE_BLACKLIST):
        return -80

    score = len(re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]", value))
    score -= len(re.findall(r"\d", value)) * 2
    score -= abs(len(value.split()) - 2) * 2

    if any(month in upper for month in MONTH_HINTS):
        score -= 12
    if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]+\s+\d{4}", value):
        score -= 24
    return score


def looks_like_measurement_amount(text: str, match: re.Match[str]) -> bool:
    normalized = normalize_text(text)
    right = normalized[match.end():match.end() + 3].upper()
    left = normalized[max(0, match.start() - 2):match.start()].upper()
    if any(right.startswith(unit) for unit in MEASUREMENT_UNITS):
        return True
    if left.endswith("/") and right[:1] in {"G", "K", "L", "M", "C"}:
        return True
    return False


def extract_segmented_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_text(entry["text"])
    matches = list(re.finditer(rf"\*?\s*({DECIMAL_AMOUNT_PATTERN.pattern})", normalized))
    if len(matches) < 2:
        return []

    items: list[dict[str, Any]] = []
    previous_end = 0
    total_length = max(len(normalized), 1)
    for match in matches:
        amount = parse_amount(match.group(0).replace("*", "").strip())
        if amount is None or amount <= 0:
            previous_end = match.end()
            continue

        candidate = normalize_product_candidate(normalized[previous_end:match.start()])
        segment_start = previous_end
        segment_end = match.end()
        previous_end = match.end()

        if product_candidate_score(candidate) < 4:
            continue

        left_ratio = max(0.0, min(1.0, segment_start / total_length))
        right_ratio = max(left_ratio, min(1.0, segment_end / total_length))
        segment_left = int(entry["left"] + (entry["width"] * left_ratio))
        segment_width = int(max(44, (entry["width"] * (right_ratio - left_ratio))))

        items.append({
            "name": candidate,
            "price": amount,
            "quantity": 1,
            "line_total": amount,
            "box_left": segment_left,
            "box_top": entry["top"],
            "box_width": segment_width,
            "box_height": max(entry["height"], 24),
        })

    return items


def polygon_to_box(points: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = int(min(xs))
    top = int(min(ys))
    width = int(max(xs) - left)
    height = int(max(ys) - top)
    return left, top, max(width, 1), max(height, 1)


def merge_lines(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    if not entries:
        return merged

    current = entries[0].copy()
    for entry in entries[1:]:
        current_bottom = current["top"] + current["height"]
        same_row = abs(entry["top"] - current["top"]) <= 7 or abs(entry["top"] - current_bottom) <= 6
        close_x = entry["left"] <= current["left"] + current["width"] + 20

        if same_row and close_x:
            current["text"] = normalize_text(f'{current["text"]} {entry["text"]}')
            right = max(current["left"] + current["width"], entry["left"] + entry["width"])
            bottom = max(current["top"] + current["height"], entry["top"] + entry["height"])
            current["left"] = min(current["left"], entry["left"])
            current["top"] = min(current["top"], entry["top"])
            current["width"] = right - current["left"]
            current["height"] = bottom - current["top"]
        else:
            merged.append(current)
            current = entry.copy()

    merged.append(current)
    return merged


def split_mixed_lines(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_entries: list[dict[str, Any]] = []
    for entry in entries:
        text = entry["text"]
        match = re.match(rf"^(?P<amount>{DECIMAL_AMOUNT_PATTERN.pattern})\s+(?P<rest>.+)$", text)
        if not match:
            split_entries.append(entry)
            continue

        amount_text = match.group("amount")
        rest_text = normalize_text(match.group("rest"))
        if not rest_text or not (rest_text[0].isdigit() or rest_text.startswith("(") or rest_text.startswith("ï¼ˆ")):
            split_entries.append(entry)
            continue

        amount_width = max(72, min(110, int(entry["width"] * 0.26)))
        split_entries.append({
            "text": amount_text,
            "left": entry["left"],
            "top": entry["top"],
            "width": amount_width,
            "height": entry["height"],
        })
        split_entries.append({
            "text": rest_text,
            "left": entry["left"] + amount_width + 8,
            "top": entry["top"],
            "width": max(entry["width"] - amount_width - 8, 40),
            "height": entry["height"],
        })

    return split_entries


def try_extract_date(lines: list[dict[str, Any]]) -> str | None:
    for line in lines:
        for match in DATE_PATTERN.finditer(line["text"]):
            candidate = match.group(1)
            candidates = [candidate]
            tail = re.split(r"[./-]", candidate)[-1]
            if len(tail) == 4:
                shortened = candidate[:-2]
                candidates.append(shortened)

            for current in candidates:
                for pattern in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"):
                    try:
                        parsed = datetime.strptime(current, pattern).date()
                        if 2018 <= parsed.year <= datetime.utcnow().year + 1:
                            return parsed.isoformat()
                    except ValueError:
                        continue
    return None


def try_extract_total(lines: list[dict[str, Any]]) -> float | None:
    priority_words = ["ODENECEK", "DAHIL TUTAR", "GENEL TOPLAM", "TOPLAM", "TUTAR"]
    for keyword in priority_words:
        for line in reversed(lines):
            upper = line["text"].upper()
            if keyword not in upper:
                continue
            if "TOPLAM KDV" in upper:
                continue
            amount = amount_from_text(line["text"])
            if amount and amount > 0:
                return amount
    for line in reversed(lines):
        amount = amount_from_text(line["text"])
        if amount and amount > 0:
            return amount
    return None


def extract_inline_item(text: str) -> tuple[str, float] | None:
    normalized = normalize_text(text)
    matches = amount_tokens(normalized)
    if not matches:
        return None

    best_item: tuple[str, float] | None = None
    best_score = -1000

    for match in reversed(matches):
        if looks_like_measurement_amount(normalized, match):
            continue

        amount = parse_amount(match.group(0))
        if amount is None or amount <= 0:
            continue

        before = normalize_product_candidate(normalized[:match.start()])
        after = normalize_product_candidate(normalized[match.end():])

        for candidate in (after, before):
            if not candidate:
                continue
            score = product_candidate_score(candidate)
            if amount > 5000:
                score -= 50
            if score > best_score:
                best_score = score
                best_item = (candidate, amount)

    return best_item if best_score >= 4 else None


def extract_items(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        current = lines[index]
        text = current["text"]

        segmented_items = extract_segmented_items(current)
        if segmented_items:
            items.extend(segmented_items)
            index += 1
            continue

        if is_code_or_meta(text):
            index += 1
            continue

        inline_item = extract_inline_item(text)
        if inline_item:
            name, inline_amount = inline_item
            if name and not is_code_or_meta(name):
                items.append({
                    "name": name,
                    "price": inline_amount,
                    "quantity": 1,
                    "line_total": inline_amount,
                    "box_left": current["left"],
                    "box_top": current["top"],
                    "box_width": current["width"],
                    "box_height": current["height"],
                })
                index += 1
                continue

        if is_product_name(text) and index + 1 < len(lines):
            next_line = lines[index + 1]
            next_amount = amount_only(next_line["text"])
            if next_amount and next_amount > 0:
                candidate = normalize_product_candidate(text)
                if product_candidate_score(candidate) >= 4:
                    items.append({
                        "name": candidate,
                        "price": next_amount,
                        "quantity": 1,
                        "line_total": next_amount,
                        "box_left": next_line["left"],
                        "box_top": next_line["top"],
                        "box_width": next_line["width"],
                        "box_height": next_line["height"],
                    })
                    index += 2
                    continue

        index += 1

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for item in items:
        item["name"] = normalize_product_candidate(item["name"])
        upper_name = item["name"].upper()
        if product_candidate_score(item["name"]) < 4:
            continue
        if re.search(r"\bADE[TL]X?\b|\bADET\b", upper_name):
            continue
        if re.fullmatch(r"(?:\d+[.,]?\d*\s*)?(?:KGX|[I1]1/KG|X\d+|TL/KG)", upper_name):
            continue
        if re.fullmatch(r"[A-Z0-9./-]{1,6}", upper_name):
            continue
        key = (upper_name, float(item["line_total"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:40]


def run_ocr(image_bytes: bytes) -> dict[str, Any]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return {"raw_text": "", "store_name": None, "receipt_date": None, "total_amount": None, "items": []}

    engine = get_ocr()
    result, _ = engine(image)
    if not result:
        return {"raw_text": "", "store_name": None, "receipt_date": None, "total_amount": None, "items": []}

    raw_entries: list[dict[str, Any]] = []
    for entry in result:
        if len(entry) < 2:
            continue
        points = entry[0]
        text = normalize_text(entry[1])
        if not text:
            continue
        left, top, width, height = polygon_to_box(points)
        raw_entries.append({
            "text": text,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })

    raw_entries.sort(key=lambda item: (item["top"], item["left"]))
    lines = split_mixed_lines(merge_lines(raw_entries))
    raw_text = "\n".join(line["text"] for line in lines)
    items = extract_items(lines)
    total_amount = try_extract_total(lines)

    if not items and total_amount and total_amount > 0:
        items = [{
            "name": "Fis Toplami",
            "price": total_amount,
            "quantity": 1,
            "line_total": total_amount,
            "box_left": max(image.shape[1] - 140, 10),
            "box_top": max(image.shape[0] - 64, 10),
            "box_width": 120,
            "box_height": 32,
        }]

    max_store_top = max(80, int(image.shape[0] * 0.22))
    store_candidates = [
        line["text"]
        for line in lines[:8]
        if line["top"] <= max_store_top and store_line_score(line["text"]) >= 8
    ]
    store_name = store_candidates[0] if store_candidates else None

    return {
        "raw_text": raw_text,
        "store_name": store_name,
        "receipt_date": try_extract_date(lines),
        "total_amount": total_amount,
        "items": items,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scan")
async def scan(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    return run_ocr(content)
