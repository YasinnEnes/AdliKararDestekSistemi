# SPRINT 10 — INITIAL AUDIT & MEVCUT DURUM RAPORU

**Proje:** FHDSS (Forensic Handwriting Decision Support System / Adli Belge ve El Yazısı Karar Destek Sistemi)  
**Kapsam:** TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı  
**Tarih:** 29 Ağustos 2026  
**Denetleyen:** Senior AI / ML & Forensic Decision-Support Systems Engineer  

---

## 1. MEVCUT ÖZELLİKLER (Tamamlanmış Modüller)

Repository genelinde yapılan derinlemesine inceleme sonucunda aşağıdaki sistem bileşenlerinin tam ve hatasız çalıştığı doğrulanmıştır:

### A. Bilgisayarlı Görü ve Ön İşleme (`cv/`)
* **Görüntü Kalite ve Adli Uygunluk Motoru (`cv/preprocessing/suitability.py`):**
  * Laplacian varyansı ile odak/hareket bulanıklığı ölçümü.
  * RMS ve Michelson kontrast metrikleri.
  * Sinyal-Gürültü Oranı (SNR dB) hesabı.
  * Mürekkep piksel yoğunluğu (*stroke density*) ve bağlı bileşen (*connected components*) analizi.
  * 4 kademeli adli uygunluk sınıflandırması (`SUITABLE`, `PARTIALLY_SUITABLE`, `INSUFFICIENT`, `NOT_COMPARABLE`).
* **Adli Görüntü İyileştirme Boru Hattı (`cv/preprocessing/pipeline.py`):**
  * Morfolojik zemin gürültüsü temizleme (Median, Gaussian, Bilateral).
  * Otsu ve Sauvola adaptif binarizasyon.
  * Radon ve Hough tabanlı eğrilik (*skew*) tespiti ve rotasyon düzeltme.
  * Guo-Hall ve Zhang-Suen 1-piksel iskeletleştirme (*skeletonization*).
* **13 Boyutlu Biyomekanik & Morfolojik Öznitelik Çıkarımı (`cv/features/extractors.py`):**
  * *Yapısal & Uzamsal:* Yazı eğimi (*slant*), temel satır eğimi ve dalgalanması (*baseline*), kapalı ilmekler (*loops*), kesişimler/kavşaklar (*crossings*), vuruş uçları (*terminals*), çizgi uzunlukları ve aralıklar.
  * *Kinematik Dinamik Proxy:* FFT spektral güç yoğunluğu ile titreme (*tremor PSD*), pürüzsüzlük enerjisi ve **2/3 Güç Yasası ($v = k \cdot \kappa^{-1/3}$)** hız/ivme dinamikleri.
* **Found & Rogers İmza Karmaşıklığı Analizi (`cv/features/complexity.py`):**
  * Dönüm noktaları (TP), kesişimler (CR), ilmekler (LP), iskelet uzunluğu (SL) ve 2B Shannon entropisi ile karmaşıklık kategorisi (`LOW`, `MEDIUM`, `HIGH`) ve taklit edilebilirlik riski.

### B. Makine Öğrenmesi, Derin Modeller ve Karar Füzyonu (`ml/`)
* **ConvNeXt-Tiny Derin Metrik Kodlayıcı (`ml/models/siamese.py`):**
  * ConvNeXt-Tiny omurgası ve 128 boyutlu $L_2$-normalize projeksiyon başlığı.
  * Deterministik çıkarım ve kosinüs benzerliği / açısal mesafe fonksiyonları (`compute_embedding_distance`).
  * ArcFace ve Supervised Contrastive Loss kayıp fonksiyonları.
* **Z-Skoru & Mahalanobis Karşılaştırma Motoru (`ml/models/comparison.py`):**
  * Referans popülasyonunun ortalama ve standart sapma dağılımı.
  * Z-skoru tabanlı adli ayrıştırma: `CONCORDANT` ($|z| \le 1.5$), `NATURAL_VARIATION` ($1.5 < |z| \le 2.5$), `SIGNIFICANT_DIFFERENCE` ($|z| > 2.5$).
* **Taklit ve Gizleme Anomali Tespiti (`ml/models/disguise.py`):**
  * Mikroskobik titreme anomalisi, 2/3 Güç Yasası kinematik kopuşu ve kasıtlı eğim değiştirme tespiti.
* **Adli Denetimli Hibrit Füzyon (`ml/models/fusion.py`):**
  * $w_{\text{morph}}=0.45, w_{\text{deep}}=0.55$ ağırlıklı baz füzyon.
  * Yapısal çelişkilerde (zıt eğim, kavşak uyuşmazlığı vb.) derin öğrenme skorunu dinamik olarak cezalandıran **Forensic Gating Mekanizması** ve nihai $S_{\text{hybrid}}$ skoru.
* **ENFSI Standartlarında Bayesyen Hipotez Değerlendirme (`ml/evaluation/hypothesis.py`):**
  * Likelihood Ratio ($LR = \frac{P(E|H_1)}{P(E|H_2)}$) ve $\log_{10}(LR)$ hesabı.
  * 9 kademeli ENFSI sözel kanıt skalası, referans sayısı ve görüntü kalitesine bağlı Belirsizlik İndeksi ($U$) ve %95 güven aralığı.
* **Çift Seviyeli Açıklanabilirlik - Dual XAI (`ml/explainability/explainer.py`):**
  * Tabüler XAI (Pozitif ve Negatif ilk 5 adli öznitelik marjinal katkısı).
  * Görsel XAI (ConvNeXt-Tiny son katmanından Grad-CAM görsel dikkat ısı haritaları).

### C. Backend API ve Bilişsel İzolasyon (`backend/`)
* **Bilişsel Yanlılık Önleme Durum Makinesi (`backend/app/core/context_isolation.py`):**
  * Phase A (Sorgulanan Belge) analiz edilip SHA-256 ile dondurulmadan Phase B'ye (Referanslar) erişimi engelleyen protokol.
* **REST API Uç Noktaları (`backend/app/api/v1/endpoints/cases.py` & `analysis.py`):**
  * Vaka CRUD, numune yükleme, uygunluk analizi, Phase A dondurma ve tam boru hattı (`/analysis/pipeline`) API'leri.
* **Adli Raporlama Servisi (`backend/app/services/report_service.py` & `report_template.html`):**
  * 18 adli bölüm içeren Jinja2 HTML ve WeasyPrint A4 PDF bilirkişi rapor çıktısı.
  * Kriptografik SHA-256 denetim izi (`audit.jsonl`).

### D. Adli İnceleme İş İstasyonu (`frontend/`)
* Next.js 14 / React tabanlı modern adli arayüz:
  * Vaka seçim ve yönetim paneli (`ExistingCasesPanel.tsx`, `CreateCasePanel.tsx`).
  * Çift kanvas görsel karşılaştırma (`DualCanvasViewer.tsx`, `MicroscopyTab.tsx`).
  * 13 morfolojik öznitelik ve karmaşıklık görünümü (`MorphologyTab.tsx`).
  * Grad-CAM ısı haritaları ve XAI paneli (`XaiTab.tsx`).
  * ENFSI Bayesyen kanıt skalası ve PDF rapor indirme (`EnfsiReportTab.tsx`).
  * Audit Trail zaman çizelgesi (`AuditTrailTimeline.tsx`).

---

## 2. EKSİK ÖZELLİKLER (Sprint 10 Hedefleri)

1. **Dataset Pipeline (`ml/datasets/`):**
   * PyTorch tabanlı adli veri seti yükleyici (`ForensicPairDataset` / `DataLoader`).
   * Yazar bazlı veri sızıntısını (*data leakage*) önleyen sıfır sızıntılı (*disjoint-writer*) Train / Val / Test bölütleme altyapısı.
   * Çift (*positive / negative / skilled forgery / disguise pairs*) üretim mekanizması.
2. **ConvNeXt-Tiny Eğitim Döngüsü (`ml/training/`):**
   * Triplet Loss ve MultiSimilarityLoss destekli PyTorch eğitim scripti (`train.py`).
   * Checkpoint kaydetme, validasyon kaybı izleme ve en iyi model ağırlıklarını `artifacts/weights/` altına kaydetme.
3. **Üç Modelin Karşılaştırmalı Benchmark Altyapısı (`ml/evaluation/`):**
   * **Model A (Deep Only):** Sadece ConvNeXt-Tiny 128D Cosine Similarity.
   * **Model B (Morphological Only):** Sadece 13 Adli Öznitelik + Mahalanobis Skoru.
   * **Model C (FHDSS Hybrid):** ConvNeXt + Morfoloji + Forensic Gating.
   * EER (*Equal Error Rate*), FAR (*False Acceptance Rate*), FRR (*False Rejection Rate*), ROC, AUC, Precision, Recall, F1 ve çıkarım gecikmesi (*latency*) hesaplayan `benchmark.py`.
4. **Eşik Değer (Threshold) & EER Analiz Modülü:**
   * Farklı threshold değerlerinde FAR ve FRR eğrilerinin çıkarılması ve EER noktasının belirlenmesi.
5. **Kapsam Koruma Mekanizması (`PsychologicalInferenceGuard`):**
   * Kişilik, karakter, ruhsal durum veya psikolojik analiz taleplerini hem backend hem API hem de frontend seviyesinde kesin olarak engelleyen ve bilimsel açıklama döndüren koruma katmanı.
6. **Etik Kapsam ve TÜBİTAK Sonuç Raporu Dokümanları (`docs/`):**
   * `docs/ETHICS_AND_SCOPE.md` (Grafoloji karşıtı bilimsel ve yasal sınırların beyanı).
   * `docs/TUBITAK_2209A_SONUC_RAPORU.md` (TÜBİTAK 2209-A formatında akademik sonuç raporu).
7. **Sprint 10 Kapsamlı Testleri:**
   * Dataset, training, benchmark, guard ve üçlü model karşılaştırma testleri.

---

## 3. KISMEN TAMAMLANMIŞ ÖZELLİKLER

* `ml/datasets/__init__.py` ve `ml/training/__init__.py` dizinleri mevcut ancak içleri boştu.
* `ml/models/siamese.py` içerisinde `SupervisedContrastiveLoss` ve `ArcFaceLoss` tanımlı ancak uçtan uca çalıştırılabilir bir `train.py` döngüsüne bağlanmamıştı.
* `ml/evaluation/hypothesis.py` Bayesyen LR hesaplamasını eksiksiz yapıyor; ancak toplu veri kümesi üzerinde FAR/FRR/EER/ROC eğrileri çizen toplu benchmark betiği (`benchmark.py`) henüz eklenmemişti.

---

## 4. TEKRAR EDİLMEMESİ GEREKEN MEVCUT MODÜLLER

Aşağıdaki modüller **mükemmel ve eksiksiz** çalışmaktadır, kesinlikle yeniden yazılmayacak veya bozulmayacaktır:
1. `cv/preprocessing/suitability.py` (Kalite değerlendirme)
2. `cv/preprocessing/pipeline.py` (Binarizasyon, skew, iskeletleştirme)
3. `cv/features/extractors.py` (13 boyutlu adli öznitelikler)
4. `cv/features/complexity.py` (Found & Rogers karmaşıklığı)
5. `ml/models/siamese.py` (`ForensicConvNeXtEncoder` mimarisi)
6. `ml/models/comparison.py` (Z-skoru motoru)
7. `ml/models/disguise.py` (Taklit tespiti)
8. `ml/models/fusion.py` (Forensic Gating hibrit füzyon)
9. `ml/explainability/explainer.py` (Grad-CAM & Tabüler XAI)
10. `ml/evaluation/hypothesis.py` (ENFSI Bayesyen LR)
11. `backend/app/core/context_isolation.py` (Bilişsel izolasyon)
12. `backend/app/services/report_service.py` (PDF ve HTML raporlama)

---

## 5. SPRINT 10 İÇİN GEREKLİ YENİ DOSYALAR

| Dosya Yolu | Amacı |
| :--- | :--- |
| `ml/datasets/forensic_dataset.py` | PyTorch Dataset, DataLoader, yazar bazlı ayrıştırma, çift üretici ve sentetik/gerçek adli veri yükleme |
| `ml/training/train.py` | ConvNeXt-Tiny için Triplet/Contrastive/ArcFace eğitim döngüsü, checkpoint kaydı ve validasyon |
| `ml/evaluation/benchmark.py` | Deep Only vs Morphological Only vs Hybrid FHDSS karşılaştırması, FAR/FRR/EER/ROC/AUC hesabı |
| `backend/app/core/guard.py` | `PsychologicalInferenceGuard` (Psikanaliz/karakter tespiti engelleme ve adli kapsam doğrulaması) |
| `docs/ETHICS_AND_SCOPE.md` | Bilimsel geçerlilik, grafoloji ayrımı, adli kısıtlar ve etik beyanname |
| `docs/TUBITAK_2209A_SONUC_RAPORU.md` | TÜBİTAK 2209-A 19 bölümlük kapsamlı proje sonuç raporu |
| `tests/test_sprint10_suite.py` | Dataset, training, benchmark üçlü karşılaştırma ve scope guard testleri |

---

## 6. POTANSİYEL BREAKING CHANGES VE ETKİ ANALİZİ

* **Mevcut API Uyumluluğu:** `api_contracts.py` üzerindeki mevcut şemalar korunacak; `PsychologicalInferenceGuard` yeni bir middleware/validator olarak eklenecek ve geçerli adli talepleri (`HANDWRITING`, `SIGNATURE`, `forensic_comparison` vb.) doğrudan geçirirken yasaklı psikanaliz isteklerini HTTP 400 ile engelleyecektir.
* **Model Çıkarım Uyumluluğu:** `ml/models/siamese.py` içerisindeki `ForensicConvNeXtEncoder` mimarisi değiştirilmeden kullanılacaktır; eğitim scripti bu mevcut sınıftan türeyecektir.
* **Gating ve Ağırlık Sabitliği:** $w_{\text{morph}}=0.45, w_{\text{deep}}=0.55$ ağırlıkları ve gating cezalandırma kuralları korunacaktır.
* **Breaking Change Riski:** **YOK (Sıfır Regresyon Riski)**.

---

## 7. ÖNCELİKLENDİRİLMİŞ GÖREV LİSTESİ

```text
[Faz 0: Initial Audit] ──► TAMAMLANDI (Bu Doküman)
         ↓
[Faz 1: Dataset Pipeline] ──► ml/datasets/forensic_dataset.py
         ↓
[Faz 2: ConvNeXt Training] ──► ml/training/train.py & artifacts/weights/
         ↓
[Faz 3 & 4: Benchmark & Karşılaştırma] ──► ml/evaluation/benchmark.py (3 Model Karşılaştırması)
         ↓
[Faz 5: Threshold & EER Analizi] ──► FAR, FRR, ROC, EER kalibrasyonu
         ↓
[Faz 6: XAI & Gating Doğrulaması] ──► Grad-CAM + Tabüler XAI + Gating teyidi
         ↓
[Faz 7: PsychologicalInferenceGuard] ──► backend/app/core/guard.py & API Entegrasyonu
         ↓
[Faz 8 & 9: Dokümantasyon] ──► docs/ETHICS_AND_SCOPE.md & TUBITAK_2209A_SONUC_RAPORU.md
         ↓
[Faz 10: Testler & Doğrulama] ──► tests/test_sprint10_suite.py
         ↓
[Faz 11: Uçtan Uca Demo & Final Rapor] ──► Final Doğrulama ve Sprint 10 Kapanış Raporu
```
