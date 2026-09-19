# FHDSS – SPRINT 18 RAPORU

**Sprint 18: Gerçek Veri Setleri, Veri Kalitesi ve Model Eğitimine Hazırlık**

**Tarih:** 16 Eylül 2026  
**Durum:** TAMAMLANDI (SUCCESS)  
**Preflight Denetimi:** PASS (13/13 Aşamalı Kritik Kontrol Başarılı)  
**Model Eğitimi Kararı:** **READY FOR REAL MODEL TRAINING**

---

## 1. Dataset Durumu (Inventory Summary)

`datasets/manifests/dataset_inventory.json` altında kaydedilen gerçek veri setlerinin durumu:

| Dataset Adı | Tip | Dil | Format | Durum | Toplam Görsel / Numune | Yazar Sayısı | Orijinal / Sahte |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CEDAR** | Çevrimdışı İmza | İngilizce | PNG | **READY** | 2,640 | 55 | 1,320 / 1,320 |
| **BHSig-260** | Çevrimdışı İmza | Bengalce & Hintçe | TIF | **WAITING (ARCHIVED_READY)** | 16,680 | 260 | 7,560 / 9,120 |
| **UTSig** | Çevrimdışı İmza | Farsça | TIF | **WAITING (ARCHIVED_READY)** | 8,280 | 115 | 3,105 / 5,175 |
| **CVL Database** | Çevrimdışı El Yazısı | Çok Dilli | JPG | **WAITING (ARCHIVED_READY)** | 996 | 27 | 996 / 0 |
| **IAM_HW** | Çevrimdışı El Yazısı Kelime | İngilizce | PNG | **WAITING (ARCHIVED_READY)** | 115,320 | 76 | 115,320 / 0 |
| **SVC2004** | Dinamik Yörünge | İngilizce & Çince | DAT | **WAITING (ARCHIVED_READY)** | 200 | 40 | 160 / 40 |

> **Not:** CEDAR veri seti `datasets/DS_Downloaded/CEDAR/CEDAR1/signatures/` altında halihazırda tam olarak açılmış ve eğitime hazır birincil adli veri kümesidir. Diğer arşivler (`BHSig-260`, `UTSig`, `CVL`, `IAM_HW`, `SVC2004`) fiziksel olarak sağlam ZIP arşivleri olarak mevcuttur ve ihtiyaç duyulduğunda `scripts/ingest_all_datasets.py` boru hattı üzerinden sıfır sızıntı ile açılabilir.

---

## 2. Veri Kalitesi ve İstatistikleri (CEDAR - Birincil Eğitim Seti)

`datasets/manifests/dataset_quality_report.json` kapsamında gerçekleştirilen denetim:
- **Taranan Toplam Görsel:** 2,640
- **Geçerli Görsel:** 2,640 (%100)
- **Bozuk / Okunamayan Görsel:** 0
- **Boş / Sabit Zeminli Görsel:** 0
- **Mükerrer (Duplicate) Görsel:** 0
- **Sıfır Boyutlu Görsel:** 0
- **Minimum Çözünürlük:** 264 × 145 piksel
- **Maksimum Çözünürlük:** 888 × 816 piksel
- **Ortalama Çözünürlük:** 543.8 × 350.5 piksel
- **Yazar Başına Numune Sayısı:** 48 numune (24 orijinal, 24 taklit/sahte)
- **P × K Sampler Uyumluluğu:** $P=4, K=4 \implies$ 16 batch boyutu için her yazarın en az 24 orijinal örneği mevcut olup pozitif çift gereksinimini fazlasıyla karşılamaktadır.

---

## 3. Split Durumu & Yazar Sızıntısı (Zero Writer Leakage)

`ml/training/dataset_split_validator.py` aracı ile CEDAR veri seti üzerinde yapılan denetim:

```text
Toplam Tekil Yazar: 55
Train Yazarları: 38 (%69.1) -> 1,824 numune
Val Yazarları: 8 (%14.5) -> 384 numune
Test Yazarları: 9 (%16.4) -> 432 numune

Train ∩ Val Kesişimi: 0
Train ∩ Test Kesişimi: 0
Val ∩ Test Kesişimi: 0
Toplam Sızan (Leaked) Yazar: 0
DURUM: PASSED (ZERO LEAKAGE)
```

Eğitim, validasyon ve test kümeleri arasında tek bir yazar dahi ortak değildir.

---

## 4. 13 Aşamalı Preflight Denetimi (`ml/training/preflight.py`)

```text
================================================================================
  FHDSS - MODEL EĞİTİMİ ÖNCESİ 13 AŞAMALI PREFLIGHT DENETİMİ (SPRINT 18.12)
================================================================================

[OK]   01. Dataset Mevcudiyeti                  : PASS - Dizin: C:\TÜBTAK\datasets\DS_Downloaded\CEDAR\CEDAR1\signatures
[OK]   02. Dataset Extraction Durumu            : PASS - full_org ve full_forg klasörleri mevcut ve erişilebilir
[OK]   03. Görsel Sayısı Kontrolü               : PASS - Toplam 2640 adli imza/el yazısı görseli yüklendi
[OK]   04. Yazar Sayısı Kontrolü                : PASS - Toplam 55 yazar tespit edildi (Minimum gereksinim: 4)
[OK]   05. Yazar Sızıntısı (Disjoint Split)     : PASS - Sıfır sızıntı doğrulandı (Train: 1824 numune, Val: 384, Test: 432)
[OK]   06. Bozuk Görsel Denetimi                : PASS - Örneklemde 0 bozuk görsel tespit edildi
[OK]   07. BalancedBatchSampler (P x K)         : PASS - Batch boyutu=16, Yazarlar=4, Pozitif çift garantisi=True
[OK]   08. Preprocessing Sözleşmesi             : PASS - Tensör biçimi=(1, 3, 224, 224), Veri tipi=torch.float32, ImageNet Norm=Aktif
[OK]   09. ArcFace Optimizer & Gradient         : PASS - requires_grad=True, gradient_norm=27.5348
[OK]   10. Açık Küme (Open-Set) Doğrulama       : PASS - EER=0.5, ROC-AUC=0.55463, Ayrık yazar çiftleri başarılı
[OK]   11. Donanım / Çalışma Ortamı             : PASS - Ortam: CPU (Standard Host) (CUDA=False, PyTorch=2.13.0+cpu)
[OK]   12. Checkpoint Dizini Erişimi            : PASS - Yazılabilir dizin: C:\TÜBTAK\artifacts\weights
[OK]   13. ONNX Export Ortamı                   : PASS - ONNX ihracı ve ONNXRuntime 1.30.0 yürütme ortamı hazır

================================================================================
  PREFLIGHT SONUCU: 13/13 KONTROL BAŞARILI
  DURUM: [PASS] - SİSTEM GERÇEK MODEL EĞİTİMİNE %100 HAZIRDIR.
================================================================================
```

---

## 5. Test Sonuçları (Tam Regresyon Doğrulaması)

- **Sprint 18 Öncesi Test Sayısı:** 134
- **Sprint 18 Yeni Eklenen Test Sayısı:** 14 ([tests/test_sprint18_pretraining_suite.py](file:///c:/TÜBTAK/tests/test_sprint18_pretraining_suite.py))
- **Toplam Test:** 148
- **Başarılı Test (PASSED):** 148 (%100)
- **Başarısız Test (FAILED):** 0
- **Frontend Derleme Durumu (Next.js 14):** 6/6 sayfa hatasız derlendi (`npm run build`).

---

## 6. Sentetik ve Gerçek Model Ayrımı

- `artifacts/weights/convnext_tiny_forensic_best.pt` sentetik checkpoint'i, model manifestosu ve ağırlık sözlüğü seviyesinde `training_type: "synthetic_test"` olarak mühürlenmiştir.
- Model sicili (`ForensicModelRegistry`), sentetik ağırlık tespit ettiğinde adli delil kararları için kullanılamayacağına dair güvenlik uyarısı üretmektedir.
- Gerçek eğitim tamamlandığında yeni ağırlıklar `training_type: "real_dataset"` olarak kaydedilecektir.

---

## 7. Gerçek Model Eğitimi Başlatma Komutu

Tüm doğrulamalar tamamlanmış olup gerçek eğitim için doğrulanmış komut hazırdır:

```bash
python ml/training/train.py \
    --data-dir datasets/DS_Downloaded/CEDAR/CEDAR1/signatures \
    --dataset-name CEDAR \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --loss-type supcon \
    --embedding-dim 128 \
    --seed 42
```

### Parametre Özeti:
- **Kullanılacak Veri Kümesi:** CEDAR (55 yazar, 2,640 imza)
- **Bölütleme Oranları:** %70 Eğitim (38 yazar), %15 Validasyon (8 yazar), %15 Test (9 yazar) - *Sıfır Yazar Sızıntılı*
- **Batch Boyutu:** 16 ($P=4$ yazar $\times$ $K=4$ numune)
- **Kayıp Fonksiyonu:** Supervised Contrastive Loss (SupCon, $\tau=0.07$) + Açık Küme EER Doğrulaması
- **Öğrenme Oranı (LR) & Optimizer:** 1e-4, AdamW, Cosine Annealing Zamanlayıcı
- **Hedef Checkpoint:** `artifacts/weights/convnext_tiny_forensic_best.pt` + `convnext_tiny_forensic.onnx`

---

## 8. KARAR

# **READY FOR REAL MODEL TRAINING**

**Gerekçe:**  
Veri seti envanteri çıkarılmış, CEDAR'ın 2,640 görselinin tamamının kalitesi doğrulanmış, yazar sızıntısı sıfırlanmış, $P \times K$ balanced batch sampler gerçek veriyle test edilmiş, deterministik tohumlama ve sentetik checkpoint ayrımı yapılmış, preflight denetimindeki 13 kontrolün 13'ü de [PASS] vermiş ve 148 testin tamamı %100 başarıyla geçmiştir.
