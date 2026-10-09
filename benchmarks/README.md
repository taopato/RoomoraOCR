# Roomora OCR Benchmark

This benchmark uses 49 deterministic samples from the MIT-licensed
`SCU-CENG/Receipt-Dataset` and one private Roomora user sample.

The image files are downloaded into `.benchmark-data/` and are not committed.

```powershell
.\.venv\Scripts\python.exe benchmarks\download_dataset.py
.\.venv\Scripts\python.exe benchmarks\evaluate.py --engine production
.\.venv\Scripts\python.exe benchmarks\evaluate.py --engine v5-latin
.\.venv\Scripts\python.exe benchmarks\evaluate.py --engine v6-small
.\.venv\Scripts\python.exe benchmarks\evaluate.py --engine v6-medium
```

To include the private A101 sample, prepare its expected JSON using the public
dataset annotation format and run:

```powershell
.\.venv\Scripts\python.exe benchmarks\download_dataset.py `
  --user-image C:\path\to\a101.jpg `
  --user-annotation C:\path\to\a101.json
```

The benchmark evaluates merchant, date, payable total, item names, item amounts,
and processing time independently. The payable total and item fields carry the
largest weight because a plausible but financially incorrect result is unsafe.

## Measured results

The deterministic 50-receipt run (49 public receipts plus the private A101
sample) on 2026-10-09 produced these results:

| Engine | Quality | Merchant | Date | Payable total | Item name F1 | Item amount | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production adaptive pipeline | 81.0% | 74.5% | 92.0% | 96.0% | 66.8% | 75.5% | 2.9 s |

The private A101 receipt scored 100% on merchant, date, payable total, item
names, and net item amounts, including its five negative discount lines.

The final row uses a strict `0.75` normalized item-name similarity threshold.
Exploratory model comparison found PP-OCRv5 Latin more accurate and roughly ten
times faster than PP-OCRv6 medium on this corpus, so v5 is the production model.

The public annotations contain known noise. For example, receipt `1787` visibly
shows a payable total of `281.40`, while its annotation says `43.37`. Reported
scores therefore remain conservative; annotation disagreements must be audited
before using this corpus to fine-tune a recognition model.
