from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Receipt OCR Service", version="3.0.0")
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

DECIMAL_AMOUNT_PATTERN = re.compile(r"(?:\d{1,3}(?:[ .,]\d{3})+|\d+)(?:[.,]\s*\d{2})(?!\d)")
SIGNED_DECIMAL_AMOUNT_PATTERN = re.compile(r"(?P<sign>[-−])\s*(?P<amount>(?:\d{1,3}(?:[ .,]\d{3})+|\d+)(?:[.,]\s*\d{2})(?!\d))")
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
    "FATURA",
    "E-ARSIV",
    "BELGE",
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
        ocr_engine = RapidOCR(params={
            "Global.log_level": "warning",
            "Det.ocr_version": OCRVersion("PP-OCRv5"),
            "Det.model_type": ModelType("mobile"),
            "Rec.ocr_version": OCRVersion("PP-OCRv5"),
            "Rec.model_type": ModelType("mobile"),
            "Rec.lang_type": LangRec("latin"),
        })
    return ocr_engine


def enhance_receipt_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrasted, (0, 0), 1.1)
    sharpened = cv2.addWeighted(contrasted, 1.55, blurred, -0.55, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def ocr_result_score(result: list[Any] | None) -> float:
    if not result:
        return 0.0

    score = 0.0
    for entry in result:
        if len(entry) < 2:
            continue
        text = normalize_text(str(entry[1]))
        confidence = float(entry[2]) if len(entry) > 2 else 0.5
        score += len(text) * max(confidence, 0.05)
        upper = text.upper()
        if any(keyword in upper for keyword in ("TOPLAM", "TUTAR", "İNDİRİM", "INDIRIM", "TARİH", "TARIH")):
            score += 10
    return score


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

    separators = [index for index, char in enumerate(value) if char in ",."]
    if separators and len(value) - separators[-1] - 1 == 2:
        decimal_index = separators[-1]
        integer_part = re.sub(r"[,.]", "", value[:decimal_index])
        decimal_part = value[decimal_index + 1:]
        value = f"{integer_part}.{decimal_part}"
    else:
        value = re.sub(r"[,.]", "", value)

    try:
        return float(value)
    except ValueError:
        return None


def product_match_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_product_candidate(value).upper())
    return re.sub(r"[^A-Z0-9]", "", normalized.encode("ascii", "ignore").decode("ascii"))


def extract_discount(text: str, assume_discount: bool = False) -> tuple[str, float] | None:
    normalized = normalize_text(text)
    matches = list(SIGNED_DECIMAL_AMOUNT_PATTERN.finditer(normalized))
    if matches:
        match = matches[-1]
        amount = parse_amount(match.group("amount"))
    elif assume_discount:
        amount_matches = amount_tokens(normalized)
        if not amount_matches:
            return None
        match = amount_matches[-1]
        amount = parse_amount(match.group(0))
    else:
        return None

    name = normalize_product_candidate(normalized[:match.start()])
    if amount is None or amount <= 0 or product_candidate_score(name) < 4:
        return None
    return name, amount


def apply_discounts(items: list[dict[str, Any]], discounts: list[tuple[str, float]]) -> None:
    for discount_name, discount_amount in discounts:
        discount_key = product_match_key(discount_name)
        if not discount_key:
            continue

        candidates = [
            item for item in items
            if product_match_key(item["name"]) == discount_key
            or product_match_key(item["name"]) in discount_key
            or discount_key in product_match_key(item["name"])
        ]
        if not candidates:
            continue

        # Repeated campaign rows on Turkish receipts generally belong to the
        # same printed product row. Keep accumulating them on that row.
        item = max(candidates, key=lambda candidate: float(candidate.get("line_total") or 0))
        original_total = float(item.get("original_line_total") or item.get("line_total") or 0)
        applied = float(item.get("discount_amount") or 0) + discount_amount
        net_total = max(0.0, round(original_total - applied, 2))
        item["original_line_total"] = original_total
        item["discount_amount"] = round(applied, 2)
        item["line_total"] = net_total
        if float(item.get("quantity") or 1) == 1:
            item["price"] = net_total


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


def contains_keyword(text: str, keyword: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(keyword)}(?![A-Z0-9])", text.upper()))


def is_code_or_meta(text: str) -> bool:
    upper = normalize_text(text).upper()
    key = product_match_key(upper)
    if not upper:
        return True
    if DATE_PATTERN.search(upper):
        return True
    if any(contains_keyword(upper, keyword) for keyword in IGNORE_KEYWORDS):
        return True
    if key == "TOP" or any(token in key for token in ("KASYER", "TOPK", "KAZANC")):
        return True
    if re.search(r"\b(?:KART|POS|TEKPOS|BANK|VISA|MASTER|MASTERCARD)\b", upper):
        return True
    if re.fullmatch(r"\d{6,}", upper):
        return True
    if re.match(r"^\d+\s*\(", upper):
        return True
    if "ADET X" in upper:
        return True
    if "TL/AD" in upper:
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
    if any(contains_keyword(upper, keyword) for keyword in STORE_BLACKLIST):
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
    if any(contains_keyword(upper, token) for token in STORE_BLACKLIST):
        return -80

    score = len(re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]", value))
    score -= len(re.findall(r"\d", value)) * 2
    score -= abs(len(value.split()) - 2) * 2

    if any(month in upper for month in MONTH_HINTS):
        score -= 12
    if any(token in product_match_key(upper) for token in ("MAGAZACILIK", "MARKET", "TICARET", "SANAYI", "LTD", "ANONIM")):
        score += 18
    if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]+\s+\d{4}", value):
        score -= 24
    return score


def looks_like_measurement_amount(text: str, match: re.Match[str]) -> bool:
    normalized = normalize_text(text)
    right = normalized[match.end():match.end() + 3].upper()
    left = normalized[max(0, match.start() - 2):match.start()].upper()
    if "%" in left:
        return True
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

    starred_matches = [match for match in matches if "*" in match.group(0)]
    if starred_matches:
        matches = starred_matches

    items: list[dict[str, Any]] = []
    previous_end = 0
    total_length = max(len(normalized), 1)
    for match in matches:
        if looks_like_measurement_amount(normalized, match):
            continue
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
    if not entries:
        return []

    rows: list[list[dict[str, Any]]] = []
    for entry in sorted(entries, key=lambda item: (item["top"] + item["height"] / 2, item["left"])):
        entry_center = entry["top"] + entry["height"] / 2
        best_row: list[dict[str, Any]] | None = None
        best_distance = float("inf")

        for row in rows[-3:]:
            row_centers = sorted(item["top"] + item["height"] / 2 for item in row)
            row_heights = sorted(item["height"] for item in row)
            row_center = row_centers[len(row_centers) // 2]
            row_height = row_heights[len(row_heights) // 2]
            center_distance = abs(entry_center - row_center)
            tolerance = max(8, min(entry["height"], row_height) * 0.55)
            if center_distance <= tolerance and center_distance < best_distance:
                best_row = row
                best_distance = center_distance

        if best_row is None:
            rows.append([entry.copy()])
        else:
            best_row.append(entry.copy())

    merged: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: item["left"])
        left = min(item["left"] for item in row)
        top = min(item["top"] for item in row)
        right = max(item["left"] + item["width"] for item in row)
        bottom = max(item["top"] + item["height"] for item in row)
        merged.append({
            "text": normalize_text(" ".join(item["text"] for item in row)),
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        })

    return sorted(merged, key=lambda item: (item["top"], item["left"]))


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


def ocr_entries_to_lines(entries: list[list[Any]]) -> list[dict[str, Any]]:
    raw_entries: list[dict[str, Any]] = []
    for entry in entries:
        if len(entry) < 2:
            continue
        text = normalize_text(str(entry[1]))
        if not text:
            continue
        left, top, width, height = polygon_to_box(entry[0])
        raw_entries.append({
            "text": text,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })

    raw_entries.sort(key=lambda item: (item["top"], item["left"]))
    return split_mixed_lines(merge_lines(raw_entries))


def receipt_candidate_score(entries: list[list[Any]]) -> float:
    lines = ocr_entries_to_lines(entries)
    if not lines:
        return 0.0

    total = try_extract_total(lines)
    items = extract_items(lines)
    score = min(ocr_result_score(entries) / 20, 25)
    score += 35 if try_extract_date(lines) else 0
    score += 80 if total else 0
    score += min(len(items), 12) * 8

    if total and items:
        item_sum = sum(float(item.get("line_total") or 0) for item in items)
        relative_error = abs(item_sum - total) / max(total, 1)
        if relative_error <= 0.03:
            score += 50
        elif relative_error <= 0.15:
            score += 20
    return score


def normalize_store_candidate(text: str) -> str:
    value = normalize_text(text)
    address_match = re.search(
        r"\b(?:MAH(?:ALLE)?|MH|CAD(?:DE)?|CD|SOK(?:AK)?|TEL|ADRES)\b|\bNO\s*:",
        value,
        flags=re.IGNORECASE,
    )
    if address_match and address_match.start() >= 5:
        value = value[:address_match.start()]
    return normalize_text(value.strip(" -*_/"))


def receipt_candidate_is_complete(entries: list[list[Any]]) -> bool:
    lines = ocr_entries_to_lines(entries)
    total = try_extract_total(lines)
    items = extract_items(lines)
    if not total or not items or not try_extract_date(lines):
        return False
    item_sum = sum(float(item.get("line_total") or 0) for item in items)
    return abs(item_sum - total) / max(total, 1) <= 0.15


def receipt_lines_look_upside_down(lines: list[dict[str, Any]]) -> bool:
    if len(lines) < 8:
        return False
    total_indices = [
        index for index, line in enumerate(lines)
        if any(token in product_match_key(line["text"]) for token in ("TOPLAM", "TOPKDV", "ODENECEK"))
    ]
    date_indices = [index for index, line in enumerate(lines) if DATE_PATTERN.search(line["text"])]
    if not total_indices:
        return False

    total_near_top = min(total_indices) <= len(lines) * 0.35
    date_near_bottom = date_indices and max(date_indices) >= len(lines) * 0.55
    top_has_store = any(store_line_score(line["text"]) >= 8 for line in lines[:8])
    bottom_has_store = any(store_line_score(line["text"]) >= 8 for line in lines[-8:])
    return total_near_top and (bool(date_near_bottom) or (not top_has_store and bottom_has_store))


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
    def positive_amounts(text: str) -> list[float]:
        return [amount for match in amount_tokens(text) if (amount := parse_amount(match.group(0))) and amount > 0]

    def is_total_label(key: str) -> bool:
        tokens = ("ODENECEK", "GENELTOPLAM", "TOPLAM", "TOPLAIM", "TOPLM", "TOPLAI")
        return any(token in key for token in tokens) or key == "TOP"

    def label_priority(index_and_line: tuple[int, dict[str, Any]]) -> int:
        key = product_match_key(index_and_line[1]["text"])
        if "ODENECEK" in key:
            return 0
        if "TOPLAMTUTAR" in key or "GENELTOPLAM" in key:
            return 1
        if ("TOPKDV" in key or key.startswith("KDVTOPLAM")) and len(positive_amounts(index_and_line[1]["text"])) >= 2:
            return 2
        if is_total_label(key) and not any(token in key for token in ("ARATOPLAM", "MALHIZMETTOPLAM")):
            return 3
        if is_total_label(key):
            return 4
        return 5

    for index, line in sorted(enumerate(lines), key=label_priority):
        key = product_match_key(line["text"])
        if "KDVDAHILTUTAR" in key or "KDVORANI" in key or "TOPLAMISKONTO" in key:
            continue

        amounts = positive_amounts(line["text"])
        has_total = is_total_label(key)
        has_tax_total = "TOPKDV" in key or key.startswith("KDVTOPLAM")
        if not has_total and not has_tax_total:
            continue
        if has_tax_total and not has_total and len(amounts) < 2:
            continue
        if has_tax_total and not has_total and len(amounts) >= 2:
            return max(amounts)

        candidates = list(amounts)
        for nearby in lines[index + 1:index + 4]:
            nearby_key = product_match_key(nearby["text"])
            nearby_amounts = positive_amounts(nearby["text"])
            is_payment = any(token in nearby_key for token in ("KREDI", "NAKIT", "BANKAKARTI", "KARTI"))
            if (not has_letters(nearby["text"]) or is_payment) and nearby_amounts:
                candidates.extend(nearby_amounts)
            if is_payment:
                break

        if candidates:
            return max(candidates)

    for line in reversed(lines):
        key = product_match_key(line["text"])
        if not any(token in key for token in ("KREDI", "NAKIT", "BANKAKARTI", "KARTI")):
            continue
        amounts = positive_amounts(line["text"])
        if amounts:
            return max(amounts)

    all_amounts = [amount for line in lines for amount in positive_amounts(line["text"])]
    if all_amounts:
        return max(all_amounts)
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
            prefix = normalized[max(0, match.start() - 3):match.start()]
            if "*" in prefix:
                score += 30
            if amount > 5000:
                score -= 50
            if score > best_score:
                best_score = score
                best_item = (candidate, amount)

    return best_item if best_score >= 4 else None


def extract_items(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    discounts: list[tuple[str, float]] = []
    in_discount_section = False
    index = 0

    while index < len(lines):
        current = lines[index]
        text = current["text"]
        upper = text.upper()

        if "INDIRIMLER" in product_match_key(upper) or "INDIRIMLERI" in product_match_key(upper):
            in_discount_section = True
            index += 1
            continue

        if in_discount_section and any(keyword in upper for keyword in ("MAL/HIZMET", "MAL HIZMET", "ÖDENECEK", "ODENECEK", "TOPKDV", "TOPLAM KDV")):
            in_discount_section = False

        discount = extract_discount(text, in_discount_section)
        if discount:
            discounts.append(discount)
            index += 1
            continue

        if in_discount_section and is_product_name(text) and index + 1 < len(lines):
            next_discount = extract_discount(f"{text} {lines[index + 1]['text']}", True)
            if next_discount:
                discounts.append(next_discount)
                index += 2
                continue

        if in_discount_section:
            index += 1
            continue

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
    for item in items:
        item["name"] = normalize_product_candidate(item["name"])
        upper_name = item["name"].upper()
        if product_candidate_score(item["name"]) < 4:
            continue
        if re.search(r"\bADE[TL]X?\b|\bADET\b", upper_name):
            continue
        if re.fullmatch(r"(?:\d+[.,]?\d*\s*)?(?:KGX|[I1]1/KG|X\d+|TL/KG)", upper_name):
            continue
        if re.fullmatch(r"[0-9./-]{1,12}", upper_name):
            continue
        duplicate = next((existing for existing in deduped if (
            existing["name"].upper() == upper_name
            and float(existing["line_total"]) == float(item["line_total"])
            and abs(int(existing.get("box_top") or 0) - int(item.get("box_top") or 0)) <= 8
        )), None)
        if duplicate:
            continue
        deduped.append(item)

    apply_discounts(deduped, discounts)
    return deduped[:40]


def run_ocr(image_bytes: bytes) -> dict[str, Any]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return {"raw_text": "", "store_name": None, "receipt_date": None, "total_amount": None, "items": []}

    def to_entries(output: Any) -> list[list[Any]]:
        boxes = output.boxes if output.boxes is not None else []
        texts = output.txts if output.txts is not None else []
        scores = output.scores if output.scores is not None else []
        return [
            [np.asarray(box).tolist(), text, float(score)]
            for box, text, score in zip(boxes, texts, scores)
        ]

    def mostly_vertical(entries: list[list[Any]]) -> bool:
        dimensions = [polygon_to_box(entry[0])[2:] for entry in entries]
        if len(dimensions) < 3:
            return False
        vertical = sum(1 for width, height in dimensions if height > width * 1.35)
        return vertical / len(dimensions) >= 0.60

    engine = get_ocr()
    original_entries = to_entries(engine(image))
    oriented_image, result = image, original_entries
    metadata_result = original_entries
    if mostly_vertical(original_entries):
        rotated_candidates = []
        for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rotated = cv2.rotate(image, rotation)
            rotated_candidates.append((rotated, to_entries(engine(rotated))))
        oriented_image, result = max(rotated_candidates, key=lambda candidate: receipt_candidate_score(candidate[1]))
        metadata_result = result
    elif not receipt_candidate_is_complete(original_entries):
        enhanced_entries = to_entries(engine(enhance_receipt_image(image)))
        if receipt_candidate_score(enhanced_entries) >= receipt_candidate_score(original_entries) + 10:
            result = enhanced_entries
    if receipt_lines_look_upside_down(ocr_entries_to_lines(result)):
        oriented_image = cv2.rotate(oriented_image, cv2.ROTATE_180)
        result = to_entries(engine(oriented_image))
        metadata_result = result

    if not result:
        return {"raw_text": "", "store_name": None, "receipt_date": None, "total_amount": None, "items": []}

    lines = ocr_entries_to_lines(result)
    raw_text = "\n".join(line["text"] for line in lines)
    items = extract_items(lines)
    total_amount = try_extract_total(lines)

    if not items and total_amount and total_amount > 0:
        items = [{
            "name": "Fis Toplami",
            "price": total_amount,
            "quantity": 1,
            "line_total": total_amount,
            "box_left": max(oriented_image.shape[1] - 140, 10),
            "box_top": max(oriented_image.shape[0] - 64, 10),
            "box_width": 120,
            "box_height": 32,
        }]

    metadata_lines = ocr_entries_to_lines(metadata_result)
    store_candidates = [
        normalize_store_candidate(line["text"])
        for line in metadata_lines[:8]
        if store_line_score(normalize_store_candidate(line["text"])) >= 8
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
