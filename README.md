# Receipt OCR Service

Bu servis ücretsiz `PaddleOCR` kullanarak fiş fotoğrafından:

- ham metin
- tarih
- toplam
- ürün kalemleri
- fiyat kutuları

çıkarmayı hedefler.

## Kurulum

```powershell
cd C:\EvArkadasimProje\Python\receipt_ocr_service
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
