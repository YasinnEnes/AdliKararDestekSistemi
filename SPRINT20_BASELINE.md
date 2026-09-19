# SPRINT 20 — INITIAL BASELINE AND HEALTH AUDIT REPORT

**Proje:** Forensic Handwriting Decision Support System (FHDSS)  
**Tarih:** 19 Eylül 2026  
**Aşama:** Sprint 20 Pre-Stabilization Baseline  
**Denetleyen:** Senior Full-Stack Engineer, ML Engineer, QA & Security Lead  

---

## 1. YÜRÜTME VE DOĞRULAMA TEST SONUÇLARI

### 1.1. Backend Test Suite (Pytest)
* **Komut:** `.venv\Scripts\python -m pytest`
* **Ortam:** Python 3.13.12, pytest-9.1.1, SQLAlchemy 2.0 (Async), PyTorch 2.13.0+cpu
* **Test Sonucu:** **150 PASSED / 150 TOTAL (%100 Başarı)**
* **Süre:** 42.54 saniye
* **Uyarılar:** 11 warnings (TorchScript exporter deprecation, TracerWarning tensor-to-bool, Starlette HTTP_413 deprecation)
* **Regresyon Durumu:** Sıfır test kırılması. Mevcut 150 testin tamamı çalışır vaziyette.

### 1.2. Frontend Build (Next.js 14)
* **Komut:** `npm run build` (`c:\TÜBTAK\frontend`)
* **Ortam:** Node.js v20+, Next.js 14.2.20, TypeScript 5.9.3
* **Build Sonucu:** **PASS (Exit Code: 0)**
* **Derlenen Rotalar:**
  * `○ /` (10.6 kB, First Load: 119 kB)
  * `○ /_not-found` (873 B, First Load: 88.1 kB)
  * `○ /admin` (8.16 kB, First Load: 114 kB)
  * `ƒ /cases/[id]` (50.4 kB, First Load: 159 kB)
  * `○ /login` (6.03 kB, First Load: 112 kB)
* **TypeScript & Lint Durumu:** 0 error. Statik ve dinamik sayfalar başarıyla derlendi.

---

## 2. MODEL VE ARTEFAKT ENVENTERİ & PARİTE ANALİZİ

### 2.1. Model Manifestosu (`artifacts/weights/model_manifest.json`)
* **Manifest Versiyonu:** 2.0.0
* **Aktif Birincil Model (`active_primary_model`):** `convnext_tiny_forensic_cedar_real`
* **Eğitim Türü:** `real_dataset` (CEDAR Adli İmza Veri Seti - 55 yazar, 2640 numune)
* **Ayrıştırma:** Sıfır yazar sızıntısı (Zero Writer Leakage - $Train \cap Val \cap Test = \emptyset$)

### 2.2. Model Ağırlıkları ve Kriptografik SHA-256 Doğrulaması

| Model Dosyası | Boyut (Byte) | Manifest SHA-256 | Disk SHA-256 | Durum | Rol |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `convnext_tiny_forensic_cedar_real_best.pt` | 336,807,225 | `b6befe7f7319c64ead170b7390eb19731101f223fc1ad91e638b227d7fce0d5f` | `b6befe7f7319c64ead170b7390eb19731101f223fc1ad91e638b227d7fce0d5f` | **DOĞRULANDI** | Gerçek Eğitilmiş Model (Primary) |
| `convnext_tiny_forensic_cedar_real.onnx` | 112,346,775 | `e1df25967b6c76d601a88c57d021fcf95d5c773e7213234d750798119d4605ce` | `e1df25967b6c76d601a88c57d021fcf95d5c773e7213234d750798119d4605ce` | **DOĞRULANDI** | Yüksek Hızlı ONNX Çıkarım |
| `convnext_tiny_forensic_best.pt` | 336,822,701 | `598b31260bcbcc2de25aa53c4837d46267a95352db1c9046ea0bb31f4b3ee553` | `598b31260bcbcc2de25aa53c4837d46267a95352db1c9046ea0bb31f4b3ee553` | **DOĞRULANDI** | Sentetik Baseline (Yalnızca Test) |

### 2.3. Model Tutarsızlığı Tespiti (Kritik Bulgu)
Manifest dosyası `convnext_tiny_forensic_cedar_real` modelini aktif birincil model olarak beyan etmesine rağmen:
* `backend/app/api/v1/endpoints/analysis.py:65`: Hardcoded olarak `convnext_tiny_forensic_best.pt` (Sentetik Model) aramaktadır.
* `backend/app/services/onnx_inference.py:40`: Hardcoded olarak `convnext_tiny_forensic.onnx` modelini varsayılan olarak başlatmaktadır.
* **Sonuç:** Üretim ve API hattında gerçek CEDAR modeli yerine sentetik test modelleri çalışmaktadır.

---

## 3. MEVCUT DURUM VE TEKNİK TESPİTLER

### 3.1. Backend Authentication Durumu
* **JWT & Refresh Token:** Backend `backend/app/core/security.py` ve `endpoints/auth.py` üzerinde 256-bit HS256 JWT ve SHA-256 hash'li `RefreshTokenEntity` rotasyon altyapısına sahiptir.
* **Güvenlik Açığı (Privilege Escalation):** `POST /api/v1/auth/register` endpoint'i kimlik doğrulaması olmadan herkese açıktır ve `payload.role` değerini (`ADMIN`) doğrudan kabul etmektedir.
* **Ghost User Açığı:** `security.py:159-169` satırlarında veritabanında silinmiş veya bulunamayan kullanıcı için token claim'lerinden sentetik `UserEntity` uydurularak yetki verilmektedir.
* **Varsayılan Parola:** `auth.py:108` satırında `password123` ile test hesabı otomatik tohumlanmaktadır.

### 3.2. Frontend Authentication & Session Durumu
* **Token Depolama:** `localStorage.getItem("fhdss_access_token")` kullanılmaktadır (XSS riski).
* **Refresh Token Entegrasyon Eksikliği:** `frontend/src/services/api.ts` Axios interceptor'ı 401 yanıtı aldığında token'ı yenilemeyi denememekte, doğrudan `localStorage` temizleyip `window.location.href = "/login"` yapmaktadır.
* **Oturum Takılması ("Güvenli adli oturum doğrulanıyor..."):**
  * `frontend/src/app/page.tsx:23` `if (loading)` durumunda bu mesajı göstermektedir.
  * `AuthContext.tsx` içerisindeki `initializeAuth` `api.getMe()` çağrısı yaparken sayfa yenilenmesinde veya API 401/timeout durumunda `window.location.href` ile `router.push` çakışmakta; `CaseContext` aynı anda `api.getCases()` 401 tetiklemekte ve UI `loading=true` durumunda kilitlenmektedir.

### 3.3. Tab State & Workflow Durumu
* **Yanlış Sekmede Kalma Kök Nedeni:**
  * `frontend/src/app/cases/[id]/page.tsx:691`: Analiz tamamlandığında (`handleRunAnalysis` / `handleProgressComplete`), kod zorla `setActiveTab("enfsi")` çağırmaktadır.
  * Kullanıcı analiz başlatırken hangi sekmede olursa olsun (`microscopy`, `morphology`, `xai` vb.) veya analiz sürerken başka sekmeye geçmiş olsa dahi zorla `enfsi` sekmesine yönlendirilmektedir.

### 3.4. Database Concurrency & N+1 Sorguları
* **N+1 Sorgusu:** `backend/app/api/v1/endpoints/cases.py:161-169` içerisindeki `list_cases` metodu her vaka için döngüde `_build_case_response` fonksiyonunu çağırarak her vaka için 2 ilave sorgu (`select(SampleEntity)...`) çalıştırmaktadır ($1 + 2N$ sorgu).
* **Vaka Numarası Yarış Durumu (Race Condition):** `cases.py:208-212` vaka numarasını `select(func.count(CaseEntity.case_id)) + 1` ile üretmektedir. Eşzamanlı vaka oluşturma isteklerinde çakışma (409 Conflict) oluşmaktadır.

---

## 4. BASELINE ÖZETİ VE HEDEF

| Bileşen | Mevcut Baseline Durumu | Sprint 20 Hedefi |
| :--- | :---: | :---: |
| **Backend Testleri** | 150 PASS / 0 FAIL | $\ge 150$ PASS (Yeni testlerle artış) |
| **Frontend Build** | PASS | PASS |
| **Model Doğruluğu** | Hardcoded sentetik model | Manifest güdümlü CEDAR modeli |
| **Session State** | Loading'de kilitlenme riski | 8 durumlu deterministik state machine |
| **Tab Kalıcılığı** | Analiz sonrası zorunlu `enfsi` sekmesi | Origin tab & kullanıcı seçimi korumalı |
| **Auth Güvenliği** | Public register ADMIN açığı, Ghost User | Korumalı register, katı DB doğrulaması |
| **Database** | N+1 sorgusu, Count+1 race condition | Eager loading (`selectinload`), Sequence/Atomic case number |
| **Frontend Test** | Yok | Vitest / RTL birim test altyapısı |
