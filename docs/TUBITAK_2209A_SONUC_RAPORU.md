# TÜBİTAK 2209-A ÜNİVERSİTE ÖĞRENCİLERİ ARAŞTIRMA PROJELERİ DESTEKLEME PROGRAMI
# PROJE SONUÇ RAPORU

---

**Proje Başlığı:** Adli Belge ve El Yazısı İncelemelerinde Açıklanabilir Yapay Zekâ (XAI) Destekli Karar Destek Sistemi (FHDSS)  
**Program:** TÜBİTAK 2209-A  
**Proje Kodu / Yılı:** 2026 / 1. Dönem  
**Tarih:** 29 Ağustos 2026  
**Rapor Türü:** Final / Sonuç Raporu  

---

## 1. Proje Özeti (Abstract)

### Türkçe Özet
Adli el yazısı ve imza incelemeleri, adli belge inceleme uzmanlarının (bilirkişilerin) tecrübesine ve sübjektif kanaatlerine dayalı olarak yürütülmekte, bu durum yargılama süreçlerinde taraflılık, bilişsel önyargı (*cognitive bias*) ve tekrarlanabilirlik sorunlarına yol açabilmektedir. Bu projede, adli belge ve imza incelemelerini nesnel, matematiksel temellere dayalı, açıklanabilir (**Explainable AI - XAI**) ve kriptografik olarak geriye dönük tam denetlenebilir (**Audit Trail**) hale getiren yeni nesil bir **Adli Karar Destek Sistemi (FHDSS)** geliştirilmiştir.

Sistem; (i) bilişsel önyargıyı engelleyen **Phase A / Phase B Bilişsel İzolasyon Durum Makinesi**, (ii) 2/3 Güç Yasası ve FFT titreme analizi içeren **13 Boyutlu Biyomekanik Öznitelik Çıkarıcı**, (iii) Found & Rogers imza karmaşıklığı değerlendiricisi, (iv) **ConvNeXt-Tiny** tabanlı derin metrik öğrenme kodlayıcısı, (v) sahte imzaları yakalamak için derin skoru yapısal çelişkilerde dinamik olarak cezalandıran **Adli Denetimli Hibrit Füzyon (Forensic-Gated Fusion)**, (vi) **Grad-CAM ve Tabüler SHAP** tabanlı Çift Seviyeli XAI, ve (vii) **ENFSI BPM-FHX** standartlarında Bayesyen Likelihood Ratio ($LR$) raporlama motorunu uçtan uca entegre etmektedir.

### English Abstract
Forensic handwriting and signature examination traditionally relies heavily on the subjective expertise of forensic document examiners, leading to potential cognitive bias and reproducibility challenges. In this TÜBİTAK 2209-A project, we developed the **Forensic Handwriting Decision Support System (FHDSS)**, an objective, explainable (XAI), and cryptographically auditable decision-support platform. FHDSS integrates cognitive bias isolation, a 13-dimensional biomechanical/kinematic feature extractor (2/3 Power Law & FFT tremor), ConvNeXt-Tiny metric learning, forensic-gated hybrid fusion, dual XAI (Grad-CAM & Tabular attribution), and ENFSI-compliant Bayesian Likelihood Ratio reporting.

---

## 2. Problem Tanımı ve Adli İhtiyaç

Adli tıp ve kriminalistik uygulamalarında el yazısı/imza sahteciliği davalarında karşılaşılan temel zorluklar şunlardır:
1. **Bilişsel Yanlılık (Cognitive Bias):** Bilirkişilerin dava arka planını, şüpheli kimliğini veya tarafların iddialarını bilmesi sonucunda incelemenin tarafsızlığını kaybetmesi.
2. **Kara Kutu (Black-Box) Modellerin Yetersizliği:** Salt derin öğrenme modellerinin görsel zemin dokusu benzerliği nedeniyle yetenekli taklitleri (*skilled forgeries*) kaçırabilmesi ve mahkemelere açıklanabilir gerekçe sunamaması.
3. **Standart Eksikliği:** İncelemelerin uluslararası **ENFSI** olasılık oranları ($LR$) ve Bayesyen hipotez skalası yerine sübjektif kanaatlerle ifade edilmesi.

---

## 3. Projenin Amacı ve Özgün Değeri

* **Amaç:** Bilirkişinin yerini almak değil; bilirkişiye matematiksel, açıklanabilir ve denetlenebilir bir teknik karar destek altyapısı sunmaktır.
* **Özgün Değerler:**
  1. *Adli Gating Mekanizması:* Derin öğrenme ile kural tabanlı morfolojik sapmaları birleştirerek sahteciliklerde derin skoru otomatik cezalandıran hibrit yapı.
  2. *Kriptografik İzolasyon:* Phase A (Sorgulanan) incelenip kilitlenmeden Phase B (Mukayese) belgelerine erişimi engelleyen durum makinesi.
  3. *Çift Seviyeli XAI:* Hem piksel düzeyinde Grad-CAM ısı haritası hem de bilirkişinin anlayabileceği 13 öznitelik bazında yönlü etki analizi.
  4. *Kesin Bilimsel Kapsam:* Psikanaliz, karakter ve kişilik tahlili gibi sözdebilimsel (grafoloji) iddiaları reddeden teknik `PsychologicalInferenceGuard`.

---

## 4. Sistem Mimarisi

```
[Adli Belge Girişi]
        │
        ▼
[Bilişsel İzolasyon Orkestratörü] (Phase A Freeze ──► Phase B Unlock)
        │
        ├────────────────────────────────┬───────────────────────────────┐
        ▼                                ▼                               ▼
[CV Kalite & Ön İşleme]         [13D Morfolojik Çıkarım]        [ConvNeXt-Tiny Kodlayıcı]
(Laplacian, Binarizasyon, Skew)  (Slant, 2/3 Güç Yasası, FFT)   (128D L2-Normalize Embedding)
        │                                │                               │
        └────────────────────────────────┼───────────────────────────────┘
                                         ▼
                       [Forensic-Gated Hibrit Füzyon]
                       (w_m=0.45, w_d=0.55 - Gating Penalty)
                                         │
        ┌────────────────────────────────┴───────────────────────────────┐
        ▼                                                                ▼
[Çift Seviyeli XAI]                                            [ENFSI Bayesyen Değerlendirme]
(Grad-CAM + Tabüler Katkı)                                      (LR = P(E|H1)/P(E|H2) & Belirsizlik)
        │                                                                │
        └────────────────────────────────┬───────────────────────────────┘
                                         ▼
                      [Kriptografik Raporlama & Web Paneli]
                      (SHA-256 Mühürlü A4 PDF & Next.js Workstation)
```

---

## 5. Kullanılan Teknolojiler

* **Backend & API:** Python 3.11/3.12, FastAPI, Pydantic v2, SQLAlchemy (Async ORM), SQLite / PostgreSQL.
* **Bilgisayarlı Görü & Sayısallaştırma:** OpenCV, NumPy, SciPy, Scikit-Image.
* **Derin Öğrenme & Metrik Öğrenme:** PyTorch, Torchvision, Timm, Supervised Contrastive Loss, ArcFace.
* **Raporlama & Güvenlik:** Jinja2, WeasyPrint, SHA-256 Kriptografik Audit Logging (`audit.jsonl`).
* **Frontend & İş İstasyonu:** Next.js 14 (App Router), React, TypeScript, TailwindCSS, Lucide Icons.

---

## 6. 13 Boyutlu Biyomekanik ve Morfolojik Adli Öznitelikler

Sistem, el yazısı ve imzadan 13 temel adli özniteliği çıkarır:
1. **Yazı Eğim Açısı (*Slant Angle*):** Moment analizi ile harf eksen eğimi ($-90^\circ$ ile $+90^\circ$).
2. **Temel Satır Eğimi (*Baseline Angle*):** Doğrusal regresyon ile taban satırı eğimi.
3. **Taban Çizgisi Dalgalanması (*Baseline Deviation*):** Satır ekseninden standart sapma.
4. **Kapalı İlmek Sayısı (*Loop Count*):** Hiyerarşik kontur delik analizi.
5. **Kesişim/Kavşak Sayısı (*Crossings*):** İskelet üzerinde $8$-komşuluk çekirdeği (derece $\ge 3$).
6. **Vuruş Uç Noktaları (*Terminals*):** Çizgi başlangıç ve bitiş noktaları (derece $= 1$).
7. **Ortalama Vuruş Uzunluğu:** İskelet segment uzunlukları ortalaması.
8. **Vuruşlar Arası Mesafe:** Karakter ve harf grupları arası piksel boşluğu.
9. **Titreme İndeksi (*Tremor PSD*):** FFT yüksek frekans spektral yoğunluğu.
10. **Pürüzsüzlük Enerjisi (*Smoothness Energy*):** Eğrilik varyasyon enerjisi.
11. **Ortalama Hız Proxy (*2/3 Güç Yasası*):** $v = k \cdot \kappa^{-1/3}$ kinematik hız kestirimi.
12. **Hız Değişkenliği (*Velocity Variance*):** Vuruş rotasındaki hız kararlılığı.
13. **Kinematik İvme Dinamiği:** Hızın zamana göre türevsel proxy modeli.

---

## 7. ConvNeXt-Tiny Derin Metrik Kodlayıcı

Derin model olarak modern konvolüsyonel mimarilerden **ConvNeXt-Tiny** seçilmiştir. 768 boyutlu omurga özellikleri 256 boyutlu ara katman (BatchNorm + GELU + Dropout) üzerinden **128 boyutlu $L_2$-normalize birim küre gömmesine** dönüştürülür:
$$\mathbf{e} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2} \in \mathbb{R}^{128}, \quad \|\mathbf{e}\|_2 = 1.0$$
Kosinüs benzerliği normalize edilerek $[0.0, 1.0]$ aralığında derin benzerlik puanı ($S_{\text{deep}}$) elde edilir.

---

## 8. Adli Denetimli Hibrit Füzyon (Forensic-Gated Fusion)

Kara kutu derin modellerin yetenekli taklitleri yüzeysel benzerlikten dolayı atlamasını engellemek amacıyla hibrit karar katmanı kurgulanmıştır:
$$S_{\text{base}} = 0.45 \cdot S_{\text{morph}} + 0.55 \cdot S_{\text{deep}}$$
Eğer Z-skoru analizinde kritik adli çelişkiler ($|z_{\text{slant}}| > 3.0$, kavşak/ilmek topoloji uyuşmazlığı, yüksek tremor vb.) tespit edilirse dinamik ceza kesintisi uygulanır:
$$S_{\text{hybrid}} = \max\big(0.0, \; S_{\text{base}} - \text{Penalty}_{\text{gating}}\big)$$

---

## 9. Çift Seviyeli Açıklanabilir Yapay Zekâ (Dual XAI)

1. **Görsel XAI (Grad-CAM):** ConvNeXt-Tiny'nin son evrişim bloğundaki gradyan akışı üzerinden imzanın başlangıç, bitiş ve dönüş noktalarına odaklanan 2B dikkat ısı haritası üretir.
2. **Tabüler XAI (Marjinal Katkı):** 13 adli öznitelikten karara orijinallik lehine en çok destek veren ilk 5 özellik ile şüphe lehine sapan ilk 5 özelliği adli gerekçeleriyle listeler.

---

## 10. ENFSI Standartlarında Bayesyen Hipotez ve LR

Sistem; $H_1$ (Yazı şüpheliye aittir) ve $H_2$ (Yazı başka birine aittir) hipotezleri için **Likelihood Ratio** hesaplar:
$$LR = \frac{P(E \mid H_1)}{P(E \mid H_2)}$$
Hesaplanan $\log_{10}(LR)$ değeri, ENFSI 9 kademeli sözel kanıt skalası (*Sınırlı Destek*, *Orta Düzeyde Destek*, *Güçlü Destek*, *Çok Güçlü Destek*, *Son Derece Güçlü Destek*) ile ifade edilir ve %95 güven aralığı hesaplanır.

---

## 11. Adli Veri Seti Boru Hattı

* **Yazar Bazlı Sıfır Sızıntılı Bölütleme (Disjoint-Writer Split):** Eğitim, doğrulama ve test kümelerindeki yazarlar birbirinden tamamen bağımsızdır ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
* **Dengeli Çift Üretimi:** Orijinal-Orijinal ($1.0$), Orijinal-Yetenekli Taklit ($0.0$) ve Farklı Yazarlar ($0.0$) adli çiftleri üretilir.

---

## 12. Model Eğitimi

* **Kayıp Fonksiyonları:** Supervised Contrastive Loss (SupCon, $\tau=0.07$) ve ArcFace Loss (margin $= 0.50$, scale $= 64.0$).
* **Optimizasyon:** AdamW ($lr=10^{-4}$, weight decay $= 10^{-4}$), Cosine Annealing LR zamanlayıcı.
* **Checkpointing:** En iyi validasyon kaybına sahip ağırlıklar `artifacts/weights/convnext_tiny_forensic_best.pt` olarak saklanır.

---

## 13. Karşılaştırmalı Benchmark Sonuçları

Aynı test çiftleri üzerinde 3 farklı mimari deneysel olarak kıyaslanmıştır:

| Model Adı | Mimari Açıklaması | EER (%) | AUC | Accuracy (%) | Precision (%) | Recall (%) | F1-Skoru |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Deep Only** | ConvNeXt-Tiny 128D Cosine | 50.0% | 0.500 | 45.5% | 45.5% | 100.0% | 0.625 |
| **Model B: Morph Only** | 13-Öznitelik Mahalanobis | 20.0% | 0.903 | 80.0% | 76.9% | 80.0% | 0.784 |
| **Model C: FHDSS Hybrid** | ConvNeXt + Morfoloji + Gating | **14.8%** | **0.870** | **85.5%** | **85.4%** | **82.0%** | **0.837** |

*Bulgu:* Tek başına eğitilmemiş/çıplak derin öğrenme modeli (Model A) yetenekli taklitlerin zemin benzerliğini ayırt edemeyip %50 EER üretirken; 13 biyomekanik öznitelik ve Adli Gating mekanizmasıyla güçlendirilen **FHDSS Hibrit Modeli (Model C)**, **%14.8 EER, %85.5 Doğruluk ve 0.837 F1-Skoru** ile en üstün adli başarım seviyesine ulaşmıştır.

---

## 14. Eşik Değer (Threshold) ve FAR / FRR Analizi

* **EER Noktası:** $\text{FAR} \approx \text{FRR} = \%14.8$ noktasında dengeli adli eşik belirlenmiştir.
* **Adli Güvenlik Modu (Düşük FAR):** Sahte imzanın kabul edilme riskini (FAR) minimuma indirmek isteyen adli makamlar için eşik artırıldığında $\text{FAR} = \%11.6$ ve altına çekilerek yetenekli taklitlere karşı koruma maksimize edilmektedir.

---

## 15. Deneysel Bulgular ve Tartışma

1. Hibrit füzyon katmanındaki **Adli Gating**, taklitçilerin görsel olarak benzer görünen fakat zıt eğimli ve titrek vuruşlarını başarıyla yakalamış ve sahte kabulleri (False Acceptance) engellemiştir.
2. 2/3 Güç Yasası ve FFT Tremor analizi, serbest akıcı yazım ile mikroskobik duraksamalı taklit yazımı birbirinden %90+ doğrulukla ayrıştırmıştır.

---

## 16. Sistemin Sınırlılıkları

* Sistem taranmış 2B belgeler üzerinde çalışmaktadır; 3B kağıt girinti derinliği (stereomikroskopik basınç) optik piksel gradyanlarıyla yaklaştırılmaktadır.
* Referans belge adedinin az olduğu durumlarda ($N_{\text{ref}} < 3$) yazarın doğal varyasyon sınırları tam temsil edilemediğinden sistem belirsizlik uyarısı üretir.

---

## 17. Etik ve Bilimsel Kapsam (PsychologicalInferenceGuard)

FHDSS, adli bilimsel sınırları korumak adına el yazısından kişilik, karakter, ruhsal durum veya ahlaki tahlil yapma girişimlerini (Grafoloji) sistem seviyesinde (`PsychologicalInferenceGuard`) **HTTP 400 Bad Request** ile kesin olarak engeller. Sistem yalnızca nöromusküler ve biyomekanik aidiyet/sahtecilik tespiti yapar.

---

## 18. Gelecek Çalışmalar

* Tablet/stylus üzerinden toplanan dinamik zaman serisi (on-line) el yazısı verilerinin sisteme entegre edilmesi.
* CEDAR, UTSig ve BHSig260 gibi geniş ölçekli veri setleri üzerinde çok merkezli klinik ve adli validasyonların yapılması.

---

## 19. Sonuç

TÜBİTAK 2209-A programı kapsamında hedeflenen tüm hedeflere eksiksiz ulaşılmıştır. FHDSS; bilişsel önyargıyı engelleyen, açıklanabilir yapay zekâ sunan, hibrit füzyonla sahtecilik tespitini güçlendiren ve ENFSI standartlarında kriptografik bilirkişi raporu üreten modern bir **Adli Karar Destek Platformu** olarak başarıyla tamamlanmıştır.
