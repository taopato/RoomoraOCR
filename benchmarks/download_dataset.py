from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "SCU-CENG/Receipt-Dataset"
BRANCH = "main"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}"
TREE_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{BRANCH}?recursive=1"
USER_AGENT = "Roomora-OCR-Benchmark/1.0"


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def fetch_json(url: str):
    return json.loads(fetch_bytes(url).decode("utf-8-sig"))


def raw_url(path: str) -> str:
    return f"{RAW_ROOT}/{urllib.parse.quote(path, safe='/')}"


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def choose_records(records: list[dict], count: int) -> list[dict]:
    buckets = {"simple": [], "medium": [], "complex": []}
    for record in records:
        item_count = len(record["expected"].get("item_list") or [])
        bucket = "simple" if item_count <= 2 else "medium" if item_count <= 6 else "complex"
        buckets[bucket].append(record)

    targets = {"simple": 14, "medium": 20, "complex": 15}
    selected: list[dict] = []
    used_companies: set[str] = set()

    for bucket, target in targets.items():
        candidates = sorted(buckets[bucket], key=lambda value: value["id"])
        diverse = []
        repeated = []
        for candidate in candidates:
            company = str(candidate["expected"].get("company") or "").casefold().strip()
            if company and company not in used_companies:
                diverse.append(candidate)
                used_companies.add(company)
            else:
                repeated.append(candidate)
        pool = diverse + repeated
        selected.extend(evenly_spaced(pool, min(target, len(pool))))

    selected_ids = {record["id"] for record in selected}
    if len(selected) < count:
        remaining = [record for record in records if record["id"] not in selected_ids]
        selected.extend(evenly_spaced(remaining, count - len(selected)))
    return sorted(selected[:count], key=lambda value: value["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a reproducible Turkish receipt benchmark.")
    parser.add_argument("--output", type=Path, default=Path(".benchmark-data"))
    parser.add_argument("--public-count", type=int, default=49)
    parser.add_argument("--candidate-count", type=int, default=180)
    parser.add_argument("--user-image", type=Path)
    parser.add_argument("--user-annotation", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    images_dir = output / "images"
    annotations_dir = output / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    tree = fetch_json(TREE_URL)["tree"]
    image_entries = {
        Path(entry["path"]).stem: entry["path"]
        for entry in tree
        if entry.get("type") == "blob" and entry["path"].startswith("Original Image/")
    }
    annotation_ids = sorted(
        Path(entry["path"]).stem
        for entry in tree
        if entry.get("type") == "blob" and entry["path"].startswith("Infromation Extraction Json/")
    )
    available_ids = [value for value in annotation_ids if value in image_entries]
    candidate_ids = evenly_spaced(available_ids, min(args.candidate_count, len(available_ids)))

    candidates = []
    for index, receipt_id in enumerate(candidate_ids, start=1):
        annotation_path = f"Infromation Extraction Json/{receipt_id}.json"
        expected = fetch_json(raw_url(annotation_path))
        candidates.append({
            "id": receipt_id,
            "source": "SCU-CENG/Receipt-Dataset",
            "image_path": image_entries[receipt_id],
            "annotation_path": annotation_path,
            "expected": expected,
        })
        print(f"annotations {index}/{len(candidate_ids)}", end="\r", flush=True)
    print()

    selected = choose_records(candidates, args.public_count)
    manifest = []
    for index, record in enumerate(selected, start=1):
        source_image = record["image_path"]
        suffix = Path(source_image).suffix.lower()
        local_image = images_dir / f"public-{record['id']}{suffix}"
        local_annotation = annotations_dir / f"public-{record['id']}.json"
        if not local_image.exists():
            local_image.write_bytes(fetch_bytes(raw_url(source_image)))
        local_annotation.write_text(
            json.dumps(record["expected"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append({
            "id": f"public-{record['id']}",
            "source": record["source"],
            "license": "MIT",
            "image": str(local_image.relative_to(output)).replace("\\", "/"),
            "annotation": str(local_annotation.relative_to(output)).replace("\\", "/"),
        })
        print(f"images {index}/{len(selected)}", end="\r", flush=True)
    print()

    if args.user_image:
        if not args.user_annotation:
            raise SystemExit("--user-annotation is required with --user-image")
        user_image = images_dir / f"user-a101{args.user_image.suffix.lower()}"
        user_annotation = annotations_dir / "user-a101.json"
        shutil.copy2(args.user_image, user_image)
        shutil.copy2(args.user_annotation, user_annotation)
        manifest.append({
            "id": "user-a101",
            "source": "Roomora user sample",
            "license": "private-test-only",
            "image": str(user_image.relative_to(output)).replace("\\", "/"),
            "annotation": str(user_annotation.relative_to(output)).replace("\\", "/"),
        })

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest)} receipts in {output}")


if __name__ == "__main__":
    main()
