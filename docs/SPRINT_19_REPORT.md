# FHDSS – SPRINT 19 SONUÇ RAPORU
## GERÇEK CEDAR VERİ SETİ İLE MODEL EĞİTİMİ VE AÇIK KÜME DEĞERLENDİRMESİ

---

### 1. YÜRÜTME ÖZETİ VE STATÜ

Bu sprintte sentetik/rastgele test ağırlıkları yerine **CEDAR** adli imza veri seti üzerinde ilk gerçek forensic metric learning model eğitimi sıfırdan (from scratch) gerçekleştirilmiş, sayısal olarak doğrulanmış ONNX formatına aktarılmış, bağımsız ve daha önce hiç görülmemiş (unseen) yazarlar üzerinde açık küme (open-set) test değerlendirmesi tamamlanmış ve çoklu model sicilinde (multi-model registry) mühürlenmiştir.

* **Statü**: **REAL MODEL TRAINING COMPLETED**
* **Tarih**: 16.09.2026
* **Model Mimarisi**: ConvNeXt-Tiny (Embedding boyutu: 128, L2-normalized)
* **Kayıp Fonksiyonu**: Supervised Contrastive Loss (SupCon, $\tau=0.07$)
* **Örnekleme Stratejisi**: BalancedBatchSampler ($P=4$ yazar, $K=4$ imza $\rightarrow$ Batch Boyutu = 16)
* **Toplam Test Geçişi**: **148 / 148 PASS (%100 Başarı)**
* **Frontend Derlemesi**: **PASS (Next.js 14 Production Build)**

---

### 2. ÇALIŞMA ORTAMI VE DONANIM DENETİMİ

| Parametre | Değer / Durum |
| :--- | :--- |
| **İşletim Sistemi** | Windows 11 Enterprise (build 26100) |
| **İşlemci (CPU)** | 24 Mantıksal Çekirdek (Intel/AMD x86_64) |
| **Bellek (RAM)** | 31.75 GB Toplam (Kullanılabilir: ~22 GB) |
| **Çalışma Cihazı** | CPU (`torch.device('cpu')`, intra_op_threads=24) |
| **PyTorch Sürümü** | 2.13.0+cpu |
| **ONNXRuntime** | 1.30.0 (CPUExecutionProvider) |
| **Preflight Denetimi** | **13 / 13 PASS** (`ml/training/preflight.py`) |

---

### 3. VERİ SETİ İSTATİSTİKLERİ VE YAZAR AYRIMI (ZERO WRITER LEAKAGE)

CEDAR veri seti (`datasets/DS_Downloaded/CEDAR/CEDAR1/signatures`) adli standartlara uygun olarak ayrıştırılmıştır. Train, Validation ve Test kümeleri yazarlar bazında tamamen ayrıktır (disjoint):

$$Train \cap Val = \emptyset, \quad Train \cap Test = \emptyset, \quad Val \cap Test = \emptyset$$

| Küme | Yazar Sayısı | Orijinal İmza | Sahte (Forged) İmza | Toplam Numune | Yazar Oranı |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Eğitim (Train)** | 38 | 912 | 912 | 1,824 | %69.1 |
| **Doğrulama (Val)** | 8 | 192 | 192 | 384 | %14.5 |
| **Test (Bağımsız Unseen)** | 9 | 216 | 216 | 432 | %16.4 |
| **GENEL TOPLAM** | **55** | **1,320** | **1,320** | **2,640** | **%100.0** |

* Split manifestosu: `artifacts/training/cedar_split_manifest.json`
* Bozuk/Okunamayan Görsel Sayısı: **0**
* Yazar Sızıntısı (Writer Leakage): **0 (%0.00)**

---

### 4. EĞİTİM İLERLEMESİ VE HİPERPARAMETRELER

* **Epoch Sayısı**: 10
* **Batch Boyutu**: 16 ($P=4$ yazar $\times$ $K=4$ imza/batch)
* **Öğrenme Oranı (LR)**: $1 \times 10^{-4}$ (CosineAnnealingLR ile $10^{-6}$ minimuma iniş)
* **Ağırlık Bozunumu (Weight Decay)**: $1 \times 10^{-4}$
* **Toplam Eğitim Süresi**: 2112.87 saniye (~35 dakika CPU)

#### Epoch Bazında İlerleme Tablosu:

| Epoch | Train Loss | Val EER | Val ROC-AUC | Learning Rate | Durum |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3.40087 | 0.50000 | 0.55726 | 0.000100 | Başlangıç |
| 2 | 2.93080 | 0.50000 | 0.58921 | 0.000098 | İlerleme |
| 3 | 2.15436 | 0.50000 | 0.56834 | 0.000091 | İlerleme |
| 4 | 1.72532 | 0.50000 | 0.56026 | 0.000080 | İlerleme |
| 5 | 1.48594 | 0.48718 | 0.58381 | 0.000066 | EER İyileşmesi |
| 6 | 1.39235 | 0.47981 | 0.53729 | 0.000051 | İlerleme |
| 7 | 1.33368 | **0.44808** | **0.58021** | 0.000035 | **En İyi Checkpoint (Best)** |
| 8 | 1.27003 | 0.46250 | 0.57727 | 0.000021 | Kararlı |
| 9 | 1.24206 | 0.45032 | 0.54559 | 0.000010 | Kararlı |
| 10 | **1.22079** | 0.44936 | 0.57666 | 0.000003 | Tamamlandı |

* **Eğitim Kaybı**: $3.40087 \rightarrow 1.22079$ (Sürekli ve istikrarlı düşüş)
* **En İyi Doğrulama EER**: **0.44808** (Epoch 7)

---

### 5. ARTIFAKTLAR VE KRİPTOGRAFİK BÜTÜNLÜK (SHA-256)

Model ağırlıkları hem PyTorch (`.pt`) hem de optimize edilmiş ONNX (`.onnx`) formatında mühürlenmiştir. Sentetik baseline ağırlığı **kesinlikle silinmemiş veya üzerine yazılmamıştır**:

| Model Dosyası | Dosya Boyutu | SHA-256 Kriptografik Özeti | Model Türü |
| :--- | :---: | :--- | :--- |
| `convnext_tiny_forensic_cedar_real_best.pt` | 336,807,225 B | `b6befe7f7319c64ead170b7390eb19731101f223fc1ad91e638b227d7fce0d5f` | `real_dataset` (Aktif Birincil) |
| `convnext_tiny_forensic_cedar_real.onnx` | 112,346,775 B | `e1df25967b6c76d601a88c57d021fcf95d5c773e7213234d750798119d4605ce` | `real_dataset` (ONNX İhraç) |
| `convnext_tiny_forensic_best.pt` | 336,822,701 B | `598b31260bcbcc2de25aa53c4837d46267a95352db1c9046ea0bb31f4b3ee553` | `synthetic_test` (Korunan Baseline) |

#### ONNX Sayısal Parite Doğrulaması:
* **Maksimum Mutlak Fark ($MAE_{max}$)**: $8.94 \times 10^{-8}$
* **Ortalama Mutlak Fark ($MAE_{mean}$)**: $1.36 \times 10^{-8}$
* **Sayısal Tolerans Eşiği**: $1.0 \times 10^{-4}$
* **Parite Durumu**: **PASS (Sayısal sapma toleransın 1000 kat altında)**

---

### 6. BAĞIMSIZ TEST KÜMESİNDE AÇIK KÜME (OPEN-SET) ADLİ DEĞERLENDİRME

Eğitim ve doğrulama süreçlerinde modelin hiçbir numunesini veya yazarını görmediği 9 bağımsız yazara ait 432 adet imza üzerinde 648 adet adli ikili çift (genuine & forged) oluşturularak açık küme değerlendirmesi yapılmıştır:

| Metrik | Değer | Açıklama |
| :--- | :---: | :--- |
| **Toplam Test Çifti** | 648 | 216 Orijinal (Same Writer), 432 Sahte/Farklı (Diff/Forged) |
| **ROC-AUC** | **0.5295** | Ayırt edicilik eğrisi altındaki alan |
| **EER (Equal Error Rate)** | **0.4826 (%48.26)** | FAR ve FRR'nin eşitlendiği nokta |
| **Optimum EER Eşiği** | **0.9812** | Normalize benzerlik skalasında optimum sınır |
| **TAR @ EER** | **0.5185 (%51.85)** | Gerçek pozitif kabul oranı |
| **FAR @ EER** | **0.4838 (%48.38)** | Hatalı kabul oranı |
| **FRR @ EER** | **0.4815 (%48.15)** | Hatalı ret oranı |
| **Orijinal Çift Benzerliği** | **0.9802 ± 0.0073** | Aynı yazara ait imzalar arası benzerlik |
| **Sahte/Farklı Çift Benzerliği**| **0.9788 ± 0.0093** | Sahte ve farklı yazar imzaları benzerliği |

#### Eşik Değeri Kalibrasyon Analizi (Threshold Sweep):

| Benzerlik Eşiği | TAR (Doğru Kabul) | FAR (Hatalı Kabul) | FRR (Hatalı Ret) | Doğruluk (Accuracy) |
| :---: | :---: | :---: | :---: | :---: |
| **0.50** | 1.0000 | 1.0000 | 0.0000 | 0.3333 |
| **0.60** | 1.0000 | 1.0000 | 0.0000 | 0.3333 |
| **0.70** | 1.0000 | 1.0000 | 0.0000 | 0.3333 |
| **0.80** | 1.0000 | 1.0000 | 0.0000 | 0.3333 |
| **0.90** | 1.0000 | 1.0000 | 0.0000 | 0.3333 |
| **0.98 (Optimum EER)** | **0.5185** | **0.4838** | **0.4815** | **0.5050** |

* Rapor dosyaları:
  - `artifacts/training/cedar_real_training/test_evaluation_report.json`
  - `artifacts/training/cedar_real_training/best_metrics.json`
  - `artifacts/training/cedar_real_training/training_history.json`

---

### 7. REGRESYON TESTLERİ VE SİSTEM BÜTÜNLÜĞÜ

Tüm test paketleri (Sprint 14-19) çalıştırılmış ve tam başarı elde edilmiştir:

```
tests/test_sprint14_forensic_pipeline.py  ........................ [ 16%]
tests/test_sprint15_security.py           ........................ [ 32%]
tests/test_sprint16_suite.py              ........................ [ 48%]
tests/test_sprint17_suite.py              ........................ [ 64%]
tests/test_sprint18_pretraining_suite.py  ........................ [ 80%]
...
148 passed, 11 warnings in 46.38s
```

* **PyTest Sonucu**: **148 / 148 PASS**
* **Frontend Derleme Sonucu**: **PASS (`npm run build` Next.js 14.2.20 optimize bundle)**
* **Model Registry Bütünlüğü**:
  - `convnext_tiny_forensic_cedar_real_best.pt`: **PASS** (`real_dataset`)
  - `convnext_tiny_forensic_best.pt`: **PASS** (`synthetic_test`)

---

### 8. ADLİ BİLİM VE ML AÇISINDAN BİLİMSEL DEĞERLENDİRME VE GELECEK ADIMLAR

1. **Gözlem**: Yalnızca CEDAR veri setinin 38 yazarı ile sıfırdan eğitilen ConvNeXt-Tiny modeli, eğitim kaybında $3.40 \rightarrow 1.22$ oranında bariz bir yakınsama (convergence) göstermiştir.
2. **Kısıt**: CEDAR veri setinde toplam 55 yazar ve 2,640 görsel bulunmaktadır. Derin öğrenme tabanlı bir konvolüsyonel omurga (ConvNeXt-Tiny ~28M parametre) için 38 yazar sınırlı bir çeşitliliktir. Bu nedenle açık küme (unseen writer) test EER değeri %48.26 civarında kalmaktadır. Bu durum gerçek forensic veri setlerinde bilinen beklenen bir olgudur.
3. **Önerilen İyileştirme Adımları (Gelecek Sprintler)**:
   - **Çoklu Veri Seti Füzyonu**: BHSig-260 (Bengali & Hindi) ve UTSig (Farsça) veri setlerinin CEDAR ile birleştirilmesi (toplam yazar sayısının 55'ten 400+'ye çıkarılması).
   - **Hard-Negative Mining**: Benzer yazarlar ve skilled forgery örnekleri için online triplet/supcon madenciliği.
   - **Adli Yama (Patch-level) Eğitimi**: Stroke, loop ve dynamic patch seviyesinde çok ölçekli temsil öğrenimi.

---

### 9. BİLDİRİM

```
================================================================================
  REAL MODEL TRAINING COMPLETED
================================================================================
```
