# Roomora OCR Service

## Docker

```bash
docker build -t roomora-ocr .
docker run --rm -p 8008:8008 roomora-ocr
```

Bu servis ücretsiz ve yerel çalışan `RapidOCR` + `PP-OCRv5 Latin mobile`
modellerini kullanarak fiş fotoğrafından:

- ham metin
- tarih
- toplam
- ürün kalemleri
- fiyat kutuları

çıkarmayı hedefler.

Görüntüler üçüncü taraf bir OCR servisine gönderilmez. Normal ve kontrastlı
görüntüler fiş alanlarının bütünlüğüne göre karşılaştırılır; yan çekilmiş
fişlerin yönü otomatik düzeltilir. Para ayrıştırma hem `1.411,00` hem de
`1,411.00` biçimini destekler ve vergi oranlarını ürün fiyatı saymaz.

## Kurulum

```powershell
cd C:\EvArkadasimProje\GitHubRepos\EvArkadasimOcr
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8008
```

## Sağlık kontrolü

```powershell
curl http://127.0.0.1:8008/health
```

Backend bu servis açıkken önce buraya gider. Servis kapalıysa mevcut OCR fallback'i devreye girer.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe benchmarks\evaluate.py --engine production
```

Benchmark verileri repoya eklenmez. Ayrıntılar için `benchmarks/README.md`
dosyasına bakın.
