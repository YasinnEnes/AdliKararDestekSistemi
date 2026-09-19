# FHDSS — Adli Bilimsel Kapsam, Etik İlkeler ve Kullanım Beyannamesi

**Proje:** Forensic Handwriting Decision Support System (FHDSS)  
**Destekleyen Program:** TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri  
**Versiyon:** 1.0 (Sprint 10)  
**Tarih:** 29 Ağustos 2026  

---

## 1. Sistemin Hukuki ve Bilimsel Niteliği

FHDSS (**Adli Belge ve El Yazısı Karar Destek Sistemi**), adli el yazısı ve imza incelemelerinde adli belge inceleme uzmanlarının (**bilirkişilerin**) karar alma süreçlerini desteklemek, incelemeleri nesnel matematiksel temellere oturtmak ve açıklanabilir yapay zekâ (**XAI**) araçlarıyla görsel kanıtlar sunmak amacıyla tasarlanmış bir **karar destek sistemidir**.

> [!IMPORTANT]
> **Nihai Adli Kanaat İnsan Bilirkişidedir:**  
> Sistem hiçbir koşulda yargı makamı veya bilirkişi yerine geçerek kesin bir aidiyet veya sahtecilik hükmü vermez. Üretilen tüm metrikler, olasılık oranları ($LR$), Z-skorları ve görsel ısı haritaları bilirkişinin teknik değerlendirmesine yardımcı olmak üzere sunulan nesnel göstergelerdir.

---

## 2. Kesin Kapsam Dışı Alanlar: Grafoloji ve Psikanaliz Yasağı

FHDSS, **Adli Belge İnceleme (Forensic Document Examination - FDE)** disiplininin katı bilimsel sınırlarına bağlı olarak geliştirilmiştir. Aşağıdaki analiz türleri **sistem seviyesinde teknik olarak engellenmiş (PsychologicalInferenceGuard)** olup kesinlikle sistemin kapsamı dışındadır:

1. **Kişilik ve Karakter Analizi Yapılmaz:** El yazısından yazarın kişiliği, dürüstlüğü, liderliği, güvenilirliği veya ahlaki vasıfları tahlil edilmez.
2. **Psikolojik ve Ruhsal Durum Çıkarımı Yapılmaz:** Yazarın anlık ruh hali (*mood*), depresyon, stres, zihinsel sağlığı veya psikolojik travmaları el yazısı geometrisinden kestirilmez.
3. **Sözdebilimsel (Pseudoscience) Grafoloji İddiaları Reddedilir:** Yazı boyutu, harf eğimi veya ilmek genişliklerinden psikolojik analiz çıkarma iddiaları (Grafoloji), modern tıp, psikoloji ve kriminalistik camiasında bilimsel geçerliliği bulunmayan sözdebilim olarak kabul edilir.

---

## 3. Uluslararası Adli Standartlar ve Yasal Uyumluluk

Sistem, adli makamlara sunulacak teknik delillerin geçerliliğini güvence altına almak için uluslararası kabul görmüş adli standartlarla uyumlu çalışır:

* **ENFSI BPM-FHX (Avrupa Adli Bilim Enstitüleri Ağı):**  
  Sistem sonuçları, ENFSI standartlarında **Bayesyen Hipotez Değerlendirmesi ($H_1 / H_2$)**, **Likelihood Ratio ($LR = P(E|H_1)/P(E|H_2)$)** ve sözel kanıt eşdeğerlik skalası ile sunulur.
* **ASTM E2290 & SWGDOC Kılavuzları:**  
  El yazısı incelemeleri yalnızca motor alışkanlıklar, hat akışkanlığı, vuruş sıralaması ve morfolojik varyasyon sınırları dahilinde değerlendirilir.
* **Daubert ve Frye Standartları:**  
  Kullanılan tüm modellerin (ConvNeXt-Tiny, 2/3 Güç Yasası hız kestirimi, Z-skoru ve Mahalanobis mesafeleri) matematiksel formülleri, hata oranları (*FAR, FRR, EER*) ve test edilebilirlikleri şeffaf biçimde raporlanır.

---

## 4. Model Belirsizliği ve Hata Oranlarının Şeffaf Sunumu

* **Skorlar Kesin Hüküm Değildir:** Benzerlik skorları ($S_{\text{hybrid}} \in [0.0, 1.0]$) istatistiksel ve biyomekanik modellemelere dayalı tahminlerdir.
* **Veri Bağımlılığı:** Benchmark sonuçları kullanılan veri setine, çözünürlüğe ve referans numune adedine ($N_{\text{ref}}$) doğrudan bağlıdır.
* **Belirsizlik İndeksi ($U \in [0, 1]$):** Referans belge sayısının yetersiz olması ($N < 3$) veya tarama kalitesinin düşüklüğü durumunda sistem otomatik olarak belirsizlik puanını artırır ve %95 güven aralığını genişleterek bilirkişiye açık uyarı verir.
* **Bilişsel Yanlılık Koruması (Cognitive Bias Isolation):** Bilirkişinin referans belgelerin varlığından etkilenmeden önce sorgulanan belgeyi nesnel incelemesini sağlamak için *Phase A* dondurulmadan *Phase B* kilitleri açılmaz.

---

## 5. Kriptografik Denetlenebilirlik (Chain of Custody)

Yapılan her inceleme, yüklenen belgelerin SHA-256 hash özetleri, kullanılan model versiyonları ve parametreleriyle birlikte zaman damgalı olarak `audit.jsonl` kütüğüne kaydedilir. Bu sayede incelemenin geçmişe dönük değiştirilemezliği ve denetlenebilirliği yasal olarak garanti edilir.
