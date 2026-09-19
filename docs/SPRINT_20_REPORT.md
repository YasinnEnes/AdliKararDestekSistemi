# Sprint 20 Report

## 1. Executive Summary

FHDSS (Forensic Handwriting Decision Support System) Sprint 20 kapsamında, sistemin mevcut temel mimarisini ve arayüz sözleşmelerini bozmadan; kullanıcı tarafından raporlanan "Güvenli adli oturum doğrulanıyor..." takılma problemi, analiz sonrası sekme sıfırlanması problemi, yapay zekâ model manifestosu ile inference kodları arasındaki ağırlık uyuşmazlığı, güvenlik açıkları (yetki yükseltme, ghost user, varsayılan kimlik bilgileri) ve veritabanı eşzamanlılık (concurrency / N+1) darboğazları tek sprint içerisinde kontrollü ve cerrahi müdahalelerle çözülmüştür.

Sprint başlangıcındaki **150 adet temel birim/entegrasyon testinin tamamı korunmuş**, üstüne eklenen **14 yeni regresyon testi** ile birlikte test sayısı **164'e yükseltilmiş** ve **%100 başarı** (`164 passed`) elde edilmiştir. Next.js 14 frontend prodüksiyon derlemesi 0 hata ile tamamlanmıştır.

---

## 2. Initial Baseline

Sprint başlangıcında yapılan durum tespitleri `SPRINT20_BASELINE.md` dosyasında belgelenmiş olup özet değerler aşağıdadır:

* **Backend Testleri:** 150 passed (42.54s).
* **Frontend Derleme:** Next.js 14.2.20 derlemesi başarılı (0 hata).
* **Model Dosyaları ve Karmaları:**
  * CEDAR Gerçek PyTorch: `convnext_tiny_forensic_cedar_real_best.pt` (SHA-256: `b6befe7f...`, Boyut: 336.8 MB) — **MEVCUT**
  * CEDAR Gerçek ONNX: `convnext_tiny_forensic_cedar_real.onnx` (SHA-256: `e1df2596...`, Boyut: 112.3 MB) — **MEVCUT**
  * Sentetik Test Baseline: `convnext_tiny_forensic_best.pt` (SHA-256: `598b3126...`, Boyut: 336.8 MB) — **MEVCUT**
* **Tespit Edilen Kök Sorunlar:**
  1. `analysis.py` ve `onnx_inference.py` hardcoded olarak sentetik baseline model dosyalarını işaret ediyordu; `model_manifest.json` üzerindeki gerçek CEDAR modeli aktif olarak kullanılmıyordu.
  2. Public `/register` endpoint'inde `role="ADMIN"` kabul ediliyordu (Privilege Escalation).
  3. `security.py` get_current_user fonksiyonunda veritabanında bulunmayan kullanıcılar için sahte/sentetik nesne üretilip yetkilendirme bypass ediliyordu (Ghost User).
  4. Varsayılan `password123` test hesabı prodüksiyonda da tetiklenebiliyordu.
  5. `CaseWorkstationPage` analiz tamamlandığında kullanıcıyı zorla `enfsi` sekmesine geçiriyor, kullanıcının çalıştığı sekme bilgisini eziyordu.
  6. Frontend oturum doğrulamasında zaman aşımı emniyeti bulunmadığından ağ kesintilerinde UI sonsuza dek "Güvenli adli oturum doğrulanıyor..." ekranında kilitleniyordu.
  7. Axios interceptor 401 durumunda token refresh yapmıyor, doğrudan sayfayı yeniliyordu. Eşzamanlı 401 isteklerinde concurrency kuyruğu yoktu.
  8. `cases.py` vaka numarası üretiminde `COUNT(*) + 1` kullanıldığından eşzamanlı vaka oluşturmada çakışma (race condition) riski mevcuttu.
  9. `list_cases` uç noktasında numuneler vaka başına ayrı SQL sorgusuyla çekildiğinden N+1 sorgu problemi mevcuttu.
  10. `docker-compose.yml` dosyasında Redis servisi ve birimler eksikti.

---

## 3. Root Cause Analysis

### 3.1 Secure Session
* **Problem:** Uygulama açılışında veya yenilendiğinde "Güvenli adli oturum doğrulanıyor..." ekranı takılı kalıyor ve yalnızca tarayıcı/uygulama kapatılıp açıldığında düzeliyordu.
* **Root Cause:** `AuthContext.tsx` içindeki `initializeAuth` akışında deterministik bir state machine bulunmuyordu. `api.getMe()` çağrısında ağ zaman aşımı emniyeti (timeout abort guard) yoktu; sunucu yanıt vermediğinde veya yavaşladığında `loading = true` sonsuza dek asılı kalıyordu. Ayrıca oturum doğrulama hatası aldığında UI deterministik bir hata kurtarma (retry / login redirect) ekranı sunmak yerine boş döngüde kalıyordu.

### 3.2 Tab State
* **Problem:** Kullanıcı analiz başlattıktan sonra analiz tamamlandığında zorla "Bilirkişi Raporu" (`enfsi`) sekmesine geçiriliyordu.
* **Root Cause:** `frontend/src/app/cases/[id]/page.tsx` dosyasında `handleJobCompleted` fonksiyonunun 691. satırında sabit `setActiveTab("enfsi")` komutu yer alıyordu. Analiz öncesi aktif sekme (`analysisOriginTabRef`) ve kullanıcının analiz sürerken kasıtlı sekme değiştirip değiştirmediği (`userInteractedTabRef`) takip edilmiyordu.

### 3.3 Model Inconsistency
* **Problem:** Manifestoda CEDAR gerçek veri setiyle eğitilmiş `convnext_tiny_forensic_cedar_real_best.pt` ve `.onnx` dosyaları tanımlı olmasına rağmen, inference uç noktaları eski sentetik modeli kullanıyordu.
* **Root Cause:** `backend/app/api/v1/endpoints/analysis.py` 80. satırında ve `backend/app/services/onnx_inference.py` 46. satırında dosya yolları hardcoded `convnext_tiny_forensic_best.pt` ve `convnext_tiny_forensic.onnx` olarak yazılmıştı. Model kayıt sicili (`ForensicModelRegistry`) tek doğruluk kaynağı (Single Source of Truth) olarak işletilmiyordu.

### 3.4 Authentication
* **Problem:** Kayıt uç noktasında rol denetimi eksikliği ve token doğrulamasında hayalet kullanıcı zafiyeti.
* **Root Cause:** 
  * `auth.py`: `/register` payload'ındaki `role` doğrudan kabul ediliyordu; harici bir kullanıcı kendini `ADMIN` yapabiliyordu.
  * `security.py`: `get_current_user` veritabanında kullanıcı bulunamadığında 401 fırlatmak yerine in-memory geçici sahte `UserEntity` üretip isteği geçiriyordu.
  * `auth.py`: `admin` ve `bilirkisi_uzman` için `password123` otomatik hesap üretimi ortam kontrolü olmadan çalışıyordu.

### 3.5 Database
* **Problem:** Vaka numarası üretiminde çakışma (race condition) ve vakalar listelenirken N+1 sorgu problemi.
* **Root Cause:** 
  * `cases.py` vaka numarası üretirken veritabanındaki vaka sayısını `COUNT(*)` ile alıp `count + 1` formatı uyguluyordu. İki eşzamanlı istek aynı sayıyı alarak 409 Conflict üretiyordu.
  * `list_cases` endpoint'inde `select(CaseEntity)` eager loading yapmadan çekiliyor, ardından her vaka için `_build_case_response` içinde `SampleEntity` sorguları tek tek çalıştırılıyordu.

### 3.6 Task Queue
* **Problem:** Sunucu yeniden başladığında kuyruktaki işlerin akıbeti belirsiz kalıyordu; SSE kopmalarında UI kilitlenebiliyordu.
* **Root Cause:** In-memory orkestratör ve iş durumları sunucu kesintisinde temiz bir kurtarma bilgisi (explicit FAILED status + recovery reason) üretmiyordu.

### 3.7 Frontend
* **Problem:** Token süresi dolduğunda veya eşzamanlı 401 isteklerinde Axios interceptor kontrolsüzce oturumu temizliyor ve sayfayı `/login`'e yönlendiriyordu.
* **Root Cause:** Interceptor'da `isRefreshing` bayrağı ve `failedQueue` kuyruk yönetimi yoktu. Refresh Token desteği olmasına rağmen interceptor tarafından otomatik yenileme tetiklenmiyordu.

---

## 4. Implemented Fixes

### Fix 1: Deterministik Session State Machine (Problem A)
* **Dosyalar:** `frontend/src/context/AuthContext.tsx`, `frontend/src/app/page.tsx`
* **Çözüm:** 
  * `SessionState` açıkça 8 duruma ayrıldı: `idle`, `initializing`, `authenticated`, `verifying`, `verified`, `expired`, `unauthorized`, `error`.
  * 8 saniyelik zaman aşımı (Timeout Guard: `Promise.race`) eklendi; ağ koptuğunda kilitlenme kökten engellendi.
  * `page.tsx` içine deterministik hata kurtarma paneli entegre edildi: Hata durumunda kullanıcıya "Tekrar Dene" ve "Giriş Ekranına Dön" seçenekleri sunulmaktadır.
  * Refresh token ile sessiz kurtarma entegre edildi.

### Fix 2: Analiz Başlatılan Sekmenin Korunması (Problem B)
* **Dosyalar:** `frontend/src/app/cases/[id]/page.tsx`
* **Çözüm:**
  * `analysisOriginTabRef` ve `userInteractedTabRef` referansları tanımlandı.
  * Analiz başlatılırken mevcut sekme `analysisOriginTabRef` içine kaydedildi.
  * Kullanıcı analiz sürerken menüden başka bir sekmeye geçerse `userInteractedTabRef.current = true` olarak işaretlendi.
  * Analiz bittiğinde (`handleJobCompleted`): Kullanıcı manuel sekme değiştirmişse tercihi korundu; aksi halde analiz öncesi sekmeye (`analysisOriginTabRef`) geri dönüldü. Hardcoded `setActiveTab("enfsi")` kaldırıldı.

### Fix 3: Model Manifestosu Tek Doğruluk Kaynağı (Problem C)
* **Dosyalar:** `ml/models/registry.py`, `backend/app/api/v1/endpoints/analysis.py`, `backend/app/services/onnx_inference.py`, `backend/app/schemas/api_contracts.py`
* **Çözüm:**
  * Hardcoded dosya yolları kaldırıldı.
  * `model_registry.get_active_pt_path()` ve `get_active_onnx_path()` manifestodaki `active_primary_model` (`convnext_tiny_forensic_cedar_real`) üzerinden dinamik çözümlendi.
  * `FullPipelineResponse` ve analiz çıktısına `model_metadata` (`model_id`, `manifest_version`, `dataset_name`, `model_sha256`) eklendi.

### Fix 4: Kimlik Doğrulama ve Rol Güvenliği (Problem D)
* **Dosyalar:** `backend/app/api/v1/endpoints/auth.py`, `backend/app/core/security.py`
* **Çözüm:**
  * Public `/register` üzerinden `role="ADMIN"` talepleri reddedilerek `EXAMINER` rolüne sanitize edildi.
  * `security.py` içinde kullanıcı veritabanında yoksa prodüksiyon/geliştirme ortamında kesin `401 Unauthorized` fırlatıldı (Ghost user engellendi).
  * `password123` varsayılan hesap oluşturma şartı kesin olarak `settings.ENVIRONMENT == "test"` kuralına bağlandı.

### Fix 5: Eşzamanlı Vaka Numarası ve N+1 Sorgu Çözümü
* **Dosyalar:** `backend/app/api/v1/endpoints/cases.py`
* **Çözüm:**
  * `_CASE_CREATION_LOCK = asyncio.Lock()` eklendi.
  * `COUNT(*) + 1` yerine veritabanındaki maksimum sıra numarası taranarak atomik artış sağlandı; çakışma durumunda döngülü retry uygulandı.
  * `list_cases` sorgusuna `selectinload(CaseEntity.samples)` eager loading uygulandı. `_build_case_response` fonksiyonunda `sa_inspect(entity).unloaded` kontrolüyle mükerrer SQL sorguları sıfırlandı.

### Fix 6: Axios 401 Concurrency Queue & Refresh Token
* **Dosyalar:** `frontend/src/services/api.ts`
* **Çözüm:**
  * `isRefreshing` ve `failedQueue` mekanizması kuruldu.
  * 401 yanıtı geldiğinde ilk istek `/auth/refresh` çağırırken, eşzamanlı gelen diğer tüm istekler kuyruğa alındı.
  * Yenilenen JWT tüm bekleyen isteklere dağıtılarak yeniden çalıştırıldı.

### Fix 7: Docker, Redis ve Yapılandırma Senkronizasyonu
* **Dosyalar:** `docker-compose.yml`, `.env.example`
* **Çözüm:**
  * `docker-compose.yml` içine `fhdss_redis` servisi ve `redis_data` volume eklendi.
  * `.env.example` içine `REDIS_URL`, `JOB_STATE_BACKEND`, `TASK_QUEUE_MAX_RETRIES`, `SIGNATURE_PROVIDER`, `TIMESTAMP_PROVIDER` parametreleri eklendi; audit log yolu `artifacts/logs/audit.jsonl` olarak eşitlendi.

---

## 5. Model Validation

Gerçekleştirilen model testleri (`MODEL-01` ... `MODEL-10`):
1. **Manifest Okunabilirliği:** `model_manifest.json` v2.0.0 başarıyla okundu.
2. **PyTorch Ağırlık Doğrulaması:** `convnext_tiny_forensic_cedar_real_best.pt` (336.8 MB) yüklendi, state_dict doğrulandı.
3. **ONNX Ağırlık Doğrulaması:** `convnext_tiny_forensic_cedar_real.onnx` (112.3 MB) ONNX Runtime oturumuyla açıldı, fallback aktifleşmedi.
4. **Sürüm Tutarlılığı:** PyTorch ve ONNX modellerinin ikisi de CEDAR gerçek veri seti mühürlü sürümünü işaret etmektedir.
5. **Sayısal Parite:** Rastgele girdi üzerinde PyTorch Encoder ve ONNX Engine L2 normalize edilmiş 128 boyutlu birim küre vektörleri üretti; tolerans doğrulandı.
6. **Determinizm:** Aynı tensör girdisi için arka arkaya yapılan çıkarımların çıktıları bit düzeyinde eşit (`np.testing.assert_array_equal`) çıktı.
7. **Hash Bütünlüğü:** `verify_weights_integrity` fonksiyonu manifestodaki SHA-256 karması ile diskteki dosya karmasının tam eşleştiğini teyit etti.
8. **Yazar Sızıntısı (Zero Writer Leakage):** `validate_writer_disjointness` fonksiyonu train/val/test ayrıklığını denetledi; kasıtlı örtüşme durumunda `ValueError` fırlatarak sızıntıyı engelledi.

---

## 6. Security Improvements

| Güvenlik Başlığı | Eski Durum | Yeni Güvenli Durum |
| :--- | :--- | :--- |
| **Public Register Role Escalation** | Dışarıdan gelen `role="ADMIN"` kabul ediliyordu. | `ADMIN` rolü filtrelendi, güvenli varsayılan `EXAMINER` atandı. |
| **Ghost User / Synthetic User Bypass** | DB'de olmayan kullanıcı için in-memory entity dönüyordu. | Kesin `401 Unauthorized` dönmektedir. |
| **Default Credentials (`password123`)** | Geliştirme/prodüksiyon modunda otomatik test hesabı açılabiliyordu. | Yalnızca `ENVIRONMENT=test` koşulunda izin verilmektedir. |
| **Token Refresh Race Condition** | Çoklu 401'de oturum düşüyordu veya mükerrer istek atılıyordu. | Concurrency kuyruğu ile tek refresh isteğiyle tüm istekler yenilendi. |
| **Global Unhandled Exceptions** | Beklenmeyen sunucu hatalarında hassas stack trace sızabiliyordu. | Global exception handler ile `request_id` içeren güvenli JSON dönmektedir. |

---

## 7. Tests Added

Yeni oluşturulan `tests/test_sprint20_regression.py` dosyası ile 14 adet kritik regresyon testi sisteme katılmıştır:

1. `test_model_01_manifest_readable_and_valid`
2. `test_model_02_manifest_pt_weights_exist_and_loadable`
3. `test_model_03_manifest_onnx_weights_exist_and_loadable`
4. `test_model_04_pt_and_onnx_version_and_metadata_consistency`
5. `test_model_05_pytorch_onnx_numerical_parity`
6. `test_model_06_inference_determinism`
7. `test_model_07_preprocessing_config_alignment`
8. `test_model_08_file_hash_verification`
9. `test_model_09_writer_leakage_detector`
10. `test_model_10_production_model_selection`
11. `test_security_registration_privilege_escalation_blocked`
12. `test_security_ghost_user_token_rejected_in_production`
13. `test_concurrency_case_number_generation_no_collisions`
14. `test_database_n_plus_one_elimination_in_list_cases`

---

## 8. Tests Before / After

| Metrik | Sprint Öncesi (Baseline) | Sprint Sonrası (Final) | Değişim |
| :--- | :---: | :---: | :---: |
| **Toplam Test Sayısı** | 150 | 164 | **+14 Yeni Test** |
| **Başarılı Testler (Passed)** | 150 | 164 | **%100 Başarı** |
| **Başarısız Testler (Failed)** | 0 | 0 | 0 |
| **Atlanan Testler (Skipped)** | 0 | 0 | 0 |
| **Test Çalışma Süresi** | 42.54s | 47.82s | Stabil |
| **Frontend Build Hataları** | 0 | 0 | Temiz |
| **TypeScript / Lint Hataları** | 0 | 0 | Temiz |

---

## 9. Regression Test Results

Çalıştırılan komut:
```bash
.venv\Scripts\python -m pytest -q
```
Çıktı Özeti:
```text
164 passed, 11 warnings in 47.82s
```

Frontend Prodüksiyon Derleme:
```bash
cd frontend && npm run build
```
Çıktı Özeti:
```text
✓ Compiled successfully
✓ Linting and checking validity of types ...
✓ Generating static pages (6/6)
✓ Finalizing page optimization ...
Route (app)                              Size     First Load JS
┌ ○ /                                    11.7 kB         120 kB
├ ○ /_not-found                          873 B          88.1 kB
├ ○ /admin                               6.06 kB         115 kB
├ ƒ /cases/[id]                          51.2 kB         160 kB
└ ○ /login                               3.86 kB         113 kB
```

---

## 10. Files Changed

* `.env.example` — Redis, providers ve audit log senkronizasyonu.
* `docker-compose.yml` — Redis servisi ve veri birimi tanımı.
* `backend/app/core/config.py` — Ayarlar ve doğrulama kuralları.
* `backend/app/core/security.py` — Ghost user açığının kapatılması.
* `backend/app/main.py` — Merkezi güvenli exception handler.
* `backend/app/api/v1/endpoints/auth.py` — Yetki yükseltme engeli ve default password güvenliği.
* `backend/app/api/v1/endpoints/cases.py` — Vaka numarası race condition ve N+1 sorgu optimizasyonu.
* `backend/app/api/v1/endpoints/analysis.py` — Model manifestosu bağlamı ve metadata dönüşü.
* `backend/app/services/onnx_inference.py` — Manifestodan dinamik aktif ONNX yolu çözme.
* `backend/app/schemas/api_contracts.py` — `model_metadata` kontrat alanı.
* `ml/models/registry.py` — `get_active_model_id`, `get_active_pt_path`, `get_active_onnx_path`, `get_active_model_metadata`.
* `frontend/src/services/api.ts` — 401 Concurrency Queue ve Refresh Token rotation interceptor'ı.
* `frontend/src/context/AuthContext.tsx` — 8 durumlu deterministik Session State Machine ve zaman aşımı emniyeti.
* `frontend/src/app/page.tsx` — Oturum kilitlenme kurtarma arayüzü ve state entegrasyonu.
* `frontend/src/app/cases/[id]/page.tsx` — Analiz öncesi sekme koruma ve kullanıcı etkileşimine saygı mekanizması.
* `tests/test_sprint20_regression.py` — 14 aşamalı kapsamlı regresyon test paketi.

---

## 11. Remaining Known Limitations

1. **Donanım Çıkarımı (Inference):** Mevcut çalışma ortamında CUDA grafik işlemcisi bulunmadığı için testler CPU üzerinde PyTorch ve ONNX Runtime (CPUExecutionProvider) ile icra edilmektedir. Prodüksiyon ortamında CUDA destekli ortamlarda hız artışı sağlanacaktır.
2. **Kuyruk Kalıcılığı:** Test ve tekil geliştirme ortamında `JOB_STATE_BACKEND="memory"` çalışmaktadır; çok sunuculu dağıtık mimariler için `docker-compose` içine eklenen Redis servisi (`JOB_STATE_BACKEND="redis"`) devreye alınmalıdır.

---

## 12. Out-of-Scope Items

Aşağıdaki başlıklar kullanıcı talimatı doğrultusunda sistem stabilitesini riske atmamak adına bu sprintin kapsamı dışında tutulmuştur:
* LayoutLM / Mask R-CNN tabanlı tam sayfa ayrıştırma modelleri.
* Fiziksel Kamu SM akıllı kart ve PKCS#11 donanım modülleri (Mock katmanı korunmuştur).
* Fiziksel RFC3161 donanım zaman damgası sunucu entegrasyonu.
* Wacom STU donanım SDK doğrudan sürücü bağlayıcıları.

---

## 13. Final Acceptance Checklist

### SESSION
* [x] "Güvenli adli oturum doğrulanıyor..." sonsuza kadar kalmıyor (8s timeout guard + retry paneli).
* [x] Tarayıcı yenilemesi sonrası oturum doğru restore ediliyor.
* [x] Logout sonrası access token, refresh token ve kullanıcı verileri temizleniyor.
* [x] Süresi dolan token için refresh token rotation devreye giriyor.
* [x] Refresh token başarıyla yeni access token üretiyor.
* [x] Eşzamanlı 401 isteklerinde tek refresh isteği atılıyor (Concurrency Queue).
* [x] Veritabanında bulunmayan kullanıcı token ile geldiğinde 401 dönüyor (Ghost user kapalı).
* [x] Stale session state riski giderildi.

### TABS
* [x] Analiz başlatılan kaynak sekme kaydediliyor (`analysisOriginTabRef`).
* [x] Analiz tamamlandığında kaynak sekme korunuyor.
* [x] Kullanıcı analiz devam ederken başka sekmeye geçerse kullanıcının tercihine saygı duyuluyor (`userInteractedTabRef`).
* [x] Analiz hatası sekme durumunu bozmuyor.
* [x] Gereksiz yere `setActiveTab("enfsi")` zorlaması kaldırıldı.

### MODEL
* [x] `model_manifest.json` tek doğruluk kaynağı (Single Source of Truth).
* [x] CEDAR gerçek veri setiyle eğitilmiş model kullanılıyor (`convnext_tiny_forensic_cedar_real_best.pt` & `.onnx`).
* [x] Sentetik test modeli prodüksiyon çıkarımında yanlışlıkla kullanılmıyor.
* [x] PyTorch model yüklemesi ve state_dict doğrulanıyor.
* [x] ONNX model oturumu ve dinamik girdi/çıktı eksenleri doğrulanıyor.
* [x] PyTorch ve ONNX sayısal çıktı parite testi geçiyor.
* [x] Ön işleme (preprocessing) konfigürasyonu eşitlendi.
* [x] Çıkarım %100 deterministik.
* [x] Yazar sızıntısı (Zero Writer Leakage) denetimi devrede.
* [x] Her analiz sonucuna model sürümü ve SHA-256 hash bilgisi mühürleniyor.

### SECURITY
* [x] Public register üzerinden `role="ADMIN"` oluşturulamıyor (Sanitize to EXAMINER).
* [x] Ghost user açığı giderildi.
* [x] `password123` otomatik hesap üretimi yalnızca test ortamında çalışıyor.
* [x] Refresh token rotation ve reuse detection aktif.
* [x] Merkezi unhandled exception handler aktif, sunucu iç bilgileri gizleniyor.

### DATABASE
* [x] `list_cases` N+1 sorgusu `selectinload` ile giderildi.
* [x] Vaka numarası race condition giderildi (Sequence tarama + Lock).
* [x] Eşzamanlı vaka oluşturma testinde 20 tekil vaka numarası 0 çakışma ile üretildi.

### TASK SYSTEM
* [x] Asenkron analiz SSE akışı ve hata durumları korundu.
* [x] Analiz sonucu hata verirse UI üzerinde doğru kullanıcı dostu mesaj gösteriliyor.

### FRONTEND
* [x] `npm run build` hatasız tamamlandı.
* [x] TypeScript ve Lint hatası bulunmuyor.
* [x] Deterministik hata kurtarma arayüzü eklendi.

### CONFIG
* [x] `docker-compose.yml` Redis servisi ve volume tanımlı.
* [x] `.env.example` `config.py` ile tam senkronize.
* [x] Audit log yolları tutarlı (`artifacts/logs/audit.jsonl`).
