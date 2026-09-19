import axios, { InternalAxiosRequestConfig } from "axios";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
export const BACKEND_URL = API_BASE_URL.replace(/\/api\/v1\/?$/, "");
export const TOKEN_STORAGE_KEY = "fhdss_access_token";
export const REFRESH_TOKEN_STORAGE_KEY = "fhdss_refresh_token";
export const USER_STORAGE_KEY = "fhdss_user";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15000,
});

// Request Interceptor: JWT Belirtecini Otomatik Ekle
apiClient.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
}, (error) => Promise.reject(error));

// Concurrency Queue for 401 Token Refresh (Sprint 20 - P1-2)
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value?: any) => void;
  reject: (reason?: any) => void;
}> = [];

const processQueue = (error: any, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

// Response Interceptor: 401 Durumunda Güvenli Refresh Token Rotation & Concurrency Queue
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    if (!originalRequest || typeof window === "undefined") {
      return Promise.reject(error);
    }

    const url = originalRequest.url || "";
    const isAuthEndpoint = url.includes("/auth/login") || url.includes("/auth/refresh");

    if (error.response?.status === 401 && !originalRequest._retry && !isAuthEndpoint) {
      const storedRefreshToken = localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);
      if (!storedRefreshToken) {
        // Refresh token yok, oturumu güvenle temizle
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
        localStorage.removeItem(USER_STORAGE_KEY);
        if (window.location.pathname !== "/login") {
          window.location.href = "/login";
        }
        return Promise.reject(error);
      }

      if (isRefreshing) {
        // Halihazırda bir refresh isteği devam ediyorsa bu isteği kuyruğa ekle
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((newToken) => {
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`;
            }
            return apiClient(originalRequest);
          })
          .catch((err) => Promise.reject(err));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        // Döngüsel interceptor tetiklenmesini önlemek için doğrudan axios ile çağır
        const res = await axios.post<{ access_token: string; refresh_token: string; token_type: string }>(
          `${API_BASE_URL}/auth/refresh`,
          { refresh_token: storedRefreshToken },
          { headers: { "Content-Type": "application/json" } }
        );

        const newAccessToken = res.data.access_token;
        const newRefreshToken = res.data.refresh_token;

        localStorage.setItem(TOKEN_STORAGE_KEY, newAccessToken);
        if (newRefreshToken) {
          localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, newRefreshToken);
        }

        apiClient.defaults.headers.common["Authorization"] = `Bearer ${newAccessToken}`;
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        }

        processQueue(null, newAccessToken);
        return apiClient(originalRequest);
      } catch (refreshErr) {
        processQueue(refreshErr, null);
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
        localStorage.removeItem(USER_STORAGE_KEY);
        if (window.location.pathname !== "/login") {
          window.location.href = "/login";
        }
        return Promise.reject(refreshErr);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

/**
 * API ve Ağ Hatalarını kullanıcı dostu güvenli Türkçe metne dönüştürür.
 * React child olarak nesne render edilip UI donmasını/kilitlenmesini kökten engeller.
 */
export function formatApiError(err: any): string {
  if (!err) return "Bilinmeyen bir hata oluştu.";

  if (err.code === "ECONNABORTED" || err.message?.toLowerCase().includes("timeout")) {
    return "Sunucu yanıt süresi aşıldı (Zaman Aşımı). Backend servisinin (FastAPI) çalıştığından emin olun.";
  }

  if (err.message === "Network Error" || !err.response) {
    return "Backend API servisine (http://localhost:8000) bağlanılamadı. Lütfen sunucunun aktif olduğunu kontrol edin.";
  }

  const detail = err.response?.data?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object") {
    if (detail.message) {
      return detail.message;
    }
    if (Array.isArray(detail)) {
      return detail.map((d: any) => d.msg || JSON.stringify(d)).join("; ");
    }
    return JSON.stringify(detail);
  }

  return err.response?.data?.message || err.message || "Sunucu işlem hatası meydana geldi.";
}

// =============================================================================
// Tip Tanımları (TypeScript Interfaces)
// =============================================================================

export type UserRole = "EXAMINER" | "SENIOR_REVIEWER" | "ADMIN";

export interface UserResponse {
  id: string;
  username: string;
  full_name?: string;
  role: UserRole;
  is_active: boolean;
  created_at_utc: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  user: UserResponse;
}

export type IsolationState =
  | "PHASE_A_INIT"
  | "PHASE_A_ANALYSIS"
  | "PHASE_A_FROZEN"
  | "PHASE_B_ACTIVE"
  | "EVALUATION_COMPLETED";

export type ExaminationType =
  | "HANDWRITING"
  | "SIGNATURE"
  | "HANDWRITING_AND_SIGNATURE"
  | "OTHER";

export type CaseStatus =
  | "DRAFT"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "ARCHIVED";

export type CaseStage =
  | "CASE_CREATED"
  | "SUITABILITY"
  | "PREPROCESSING"
  | "FEATURE_EXTRACTION"
  | "COMPARISON"
  | "COMPLEXITY_ANALYSIS"
  | "SIMULATION_ANALYSIS"
  | "EVALUATION"
  | "REPORT"
  | "COMPLETED";

export interface CaseResponse {
  case_id: string;
  name: string;
  description: string;
  examination_type: ExaminationType;
  reference_number?: string;
  authority?: string;
  examiner_id: string;
  notes?: string;
  status: CaseStatus;
  current_stage: CaseStage;
  state: IsolationState;
  questioned_sample_id?: string;
  reference_sample_ids: string[];
  created_at_utc: string;
  updated_at_utc: string;
}

export interface CaseCreatePayload {
  name: string;
  description: string;
  examination_type: ExaminationType;
  case_id?: string;
  reference_number?: string;
  authority?: string;
  examiner_id?: string;
  notes?: string;
  status?: CaseStatus;
  current_stage?: CaseStage;
}

export interface CaseUpdatePayload {
  name?: string;
  description?: string;
  examination_type?: ExaminationType;
  reference_number?: string;
  authority?: string;
  examiner_id?: string;
  notes?: string;
  status?: CaseStatus;
  current_stage?: CaseStage;
}

export interface SampleUploadResponse {
  sample_id: string;
  case_id: string;
  filename: string;
  sample_type: "questioned" | "reference";
  sha256_hash: string;
  file_size_bytes: number;
}

export interface SuitabilityMetrics {
  blur_laplacian_var: number;
  rms_contrast: number;
  michelson_contrast: number;
  snr_db: number;
  stroke_density: number;
  connected_components_count: number;
  width: number;
  height: number;
}

export interface SuitabilityResult {
  status: "SUITABLE" | "PARTIALLY_SUITABLE" | "INSUFFICIENT" | "NOT_COMPARABLE";
  overall_score: number;
  is_admissible: boolean;
  metrics: SuitabilityMetrics;
  reasons: string[];
  recommendations: string[];
}

export interface FeatureDifferenceDetail {
  feature_key: string;
  feature_name_tr: string;
  questioned_value: number;
  reference_mean: number;
  reference_std: number;
  absolute_difference: number;
  z_score: number;
  diff_type: "CONCORDANT" | "NATURAL_VARIATION" | "SIGNIFICANT_DIFFERENCE";
  explanation: string;
}

export interface ComparisonReport {
  similarity_score: number;
  euclidean_distance: number;
  mahalanobis_distance: number;
  concordant_features: FeatureDifferenceDetail[];
  natural_variations: FeatureDifferenceDetail[];
  significant_differences: FeatureDifferenceDetail[];
  preliminary_opinion: string;
  reference_count: number;
  audit_hash_sha256: string;
}

export interface ComplexityResult {
  overall_complexity_index: number;
  category: "LOW" | "MEDIUM" | "HIGH";
  components: {
    turning_points: number;
    crossings: number;
    loops: number;
    skeleton_length_px: number;
    spatial_entropy: number;
  };
  forgery_vulnerability: string;
  confidence: number;
}

export interface SimulationIndicators {
  indicator_level: "NO_STRONG_EVIDENCE" | "POSSIBLE_INDICATORS" | "STRONG_INDICATORS";
  overall_anomaly_score: number;
  tremor_anomaly_detected: boolean;
  kinematic_decoupling_detected: boolean;
  structural_disguise_detected: boolean;
  anomaly_details: string[];
  confidence: number;
  audit_hash_sha256: string;
}

export interface HybridFusionResult {
  hybrid_similarity_score: number;
  morphological_similarity: number;
  deep_metric_similarity: number;
  gating_penalty: number;
  gating_triggered: boolean;
  concordance_level: string;
  forensic_explanation: string;
  audit_hash_sha256: string;
  raw_hybrid_score?: number;
  calibration_status?: string;
  penalty_details?: string[];
}

export interface TabularFeatureAttribution {
  feature_key: string;
  feature_name_tr: string;
  impact_score: number;
  direction: "POSITIVE" | "NEGATIVE";
  description: string;
}

export interface DualExplanationReport {
  top_positive_features: TabularFeatureAttribution[];
  top_negative_features: TabularFeatureAttribution[];
  visual_attention_summary: string;
  audit_hash_sha256: string;
}

export interface ForensicSimilarityScaleEvaluation {
  band:
    | "SUPERIMPOSITION_ANOMALY"
    | "NATURAL_VARIATION_MATCH"
    | "SKILLED_SIMULATION_ZONE"
    | "INCONCLUSIVE_DISGUISE_ZONE"
    | "LOW_SIMILARITY_NON_SOURCE"
    | "COMPLETE_DISPARITY";
  band_label_tr: string;
  similarity_percentage: number;
  min_pct: number;
  max_pct: number;
  risk_category: string;
  forensic_interpretation_tr: string;
  is_superimposition_anomaly: boolean;
  requires_micro_inspection: boolean;
}

export interface HypothesisEvaluationReport {
  support_level: string;
  log_likelihood_ratio: number;
  likelihood_ratio_estimate: number;
  uncertainty_score: number;
  confidence_interval_log_lr: [number, number];
  propositions: Record<string, string>;
  verbal_conclusion_tr: string;
  forensic_limitations: string[];
  similarity_scale_evaluation?: ForensicSimilarityScaleEvaluation;
  audit_hash_sha256: string;
}

export interface FullPipelineResponse {
  case_id: string;
  suitability: SuitabilityResult;
  complexity: ComplexityResult;
  comparison: ComparisonReport;
  disguise: SimulationIndicators;
  fusion: HybridFusionResult;
  xai: DualExplanationReport;
  hypothesis: HypothesisEvaluationReport;
  report_paths: {
    html_path?: string;
    pdf_path?: string;
    report_sha256?: string;
  };
  audit_checksum_sha256: string;
  merkle_root_hash?: string;
}

export interface CounterfactualPerturbation {
  feature_key: string;
  feature_name_tr: string;
  current_value: number;
  simulated_value: number;
  absolute_change: number;
  original_z_score: number;
  simulated_z_score: number;
  explanation_tr: string;
}

export interface CounterfactualScenario {
  scenario_id: string;
  title: string;
  target_goal: string;
  original_hybrid_score: number;
  simulated_hybrid_score: number;
  score_delta: number;
  original_support_scale: string;
  simulated_support_scale: string;
  gating_penalty_removed: number;
  required_perturbations: CounterfactualPerturbation[];
  feasibility_score: number;
}

export interface CounterfactualResponse {
  case_id: string;
  original_score: number;
  simulated_hybrid_score: number;
  simulated_morphological_score: number;
  gating_penalty: number;
  gating_triggered: boolean;
  structural_contradictions: string[];
  concordance_level: string;
  sensitivity_matrix: Record<string, number>;
  top_scenarios: CounterfactualScenario[];
  execution_time_ms: number;
}

export interface BiometricProfile {
  duration_sec: number;
  total_points: number;
  pen_down_points: number;
  pen_up_points: number;
  in_air_time_ratio: number;
  mean_velocity: number;
  max_velocity: number;
  velocity_std: number;
  mean_acceleration: number;
  max_acceleration: number;
  mean_jerk: number;
  mean_pressure: number;
  pressure_variance: number;
  pressure_peaks_count: number;
  kinematic_smoothness: number;
}

export interface BiometricCompareResponse {
  dtw_distance: number;
  dtw_normalized_similarity: number;
  velocity_correlation: number;
  pressure_correlation: number;
  in_air_time_diff: number;
  is_biometrically_consistent: boolean;
  verbal_assessment_tr: string;
  questioned_profile: BiometricProfile;
  reference_profile: BiometricProfile;
}

// =============================================================================
// API İstemci Fonksiyonları
// =============================================================================

export const api = {
  // Kimlik Doğrulama & Kullanıcı
  async login(username: string, password: string): Promise<TokenResponse> {
    const res = await apiClient.post<TokenResponse>("/auth/login", { username, password });
    return res.data;
  },

  async getMe(): Promise<UserResponse> {
    const res = await apiClient.get<UserResponse>("/auth/me");
    return res.data;
  },

  async register(payload: { username: string; password: string; full_name?: string; role?: string }): Promise<UserResponse> {
    const res = await apiClient.post<UserResponse>("/auth/register", payload);
    return res.data;
  },

  // Sistem Sağlık
  async getHealth() {
    const res = await apiClient.get("/health");
    return res.data;
  },

  // Vaka Yönetimi
  async getCases() {
    const res = await apiClient.get<CaseResponse[]>("/cases");
    return res.data;
  },

  async createCase(payload: CaseCreatePayload) {
    const res = await apiClient.post<CaseResponse>("/cases", payload);
    return res.data;
  },

  async getCase(caseId: string) {
    const res = await apiClient.get<CaseResponse>(`/cases/${caseId}`);
    return res.data;
  },

  async updateCase(caseId: string, payload: CaseUpdatePayload) {
    const res = await apiClient.put<CaseResponse>(`/cases/${caseId}`, payload);
    return res.data;
  },

  async deleteCase(caseId: string) {
    const res = await apiClient.delete<{ status: string; message: string; case_id: string }>(`/cases/${caseId}`);
    return res.data;
  },

  async uploadSample(caseId: string, file: File, sampleType: "questioned" | "reference", sampleId?: string) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sample_type", sampleType);
    if (sampleId) formData.append("sample_id", sampleId);

    const res = await apiClient.post<SampleUploadResponse>(`/cases/${caseId}/samples`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  async deleteSample(caseId: string, sampleId: string) {
    const res = await apiClient.delete<{ status: string; message: string; case_id: string; sample_id: string }>(
      `/cases/${caseId}/samples/${sampleId}`
    );
    return res.data;
  },

  // Uygunluk Değerlendirmesi
  async assessSuitability(file?: File, sampleId?: string) {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiClient.post("/analysis/suitability", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return res.data;
    } else {
      const formData = new FormData();
      if (sampleId) formData.append("sample_id", sampleId);
      const res = await apiClient.post("/analysis/suitability", formData);
      return res.data;
    }
  },

  // Bilişsel İzolasyon - Phase A Dondurma
  async isolatePhaseA(caseId: string, questionedSampleId: string, notes?: string) {
    const res = await apiClient.post("/analysis/isolate-phase-a", {
      case_id: caseId,
      questioned_sample_id: questionedSampleId,
      examiner_notes: notes || "",
    });
    return res.data;
  },

  // Tam Pipeline Çalıştır
  async runFullPipeline(payload: {
    case_id: string;
    questioned_sample_id: string;
    reference_sample_ids: string[];
    suspect_name?: string;
    generate_pdf?: boolean;
  }) {
    const res = await apiClient.post<FullPipelineResponse>("/analysis/full-pipeline", payload);
    return res.data;
  },

  // Vakaya Ait En Son Analiz Sonucunu Getir
  async getLatestAnalysisResult(caseId: string): Promise<FullPipelineResponse> {
    const res = await apiClient.get<FullPipelineResponse>(`/analysis/result/${caseId}`);
    return res.data;
  },

  // Karşıtlı Açıklama (Counterfactual XAI) Simülasyonu
  async simulateCounterfactual(payload: {
    case_id: string;
    custom_perturbations?: Record<string, number>;
    deep_similarity_score?: number;
  }) {
    const res = await apiClient.post<CounterfactualResponse>("/analysis/counterfactual", payload);
    return res.data;
  },

  // Biyometrik İmza Karşılaştırma & Demo
  async getBiometricDemo() {
    const res = await apiClient.get<BiometricCompareResponse>("/biometrics/synthetic-demo");
    return res.data;
  },

  async compareBiometrics(payload: {
    case_id?: string;
    questioned_points: any[];
    reference_points: any[];
  }) {
    const res = await apiClient.post<BiometricCompareResponse>("/biometrics/compare", payload);
    return res.data;
  },

  // Rapor PDF İndir URL
  getReportPdfUrl(caseId: string) {
    return `${API_BASE_URL}/analysis/report/pdf/${caseId}`;
  },

  // Denetim İzi (Audit Trail)
  async getCaseAuditLogs(caseId: string) {
    const res = await apiClient.get<{
      case_id: string;
      count: number;
      audit_trail: Array<{
        id: string;
        event_type: string;
        action: string;
        examiner_id?: string;
        sha256_hash: string;
        previous_hash?: string;
        sequence_number?: number;
        details?: any;
        created_at_utc: string;
      }>;
    }>(`/audit/case/${caseId}`);
    return res.data;
  },

  // Arşiv İhraç ve Doğrulama
  async exportArchive(caseId: string) {
    const res = await apiClient.post<{
      case_id: string;
      filename: string;
      download_url: string;
      size_bytes: number;
      message: string;
    }>(`/archive/export/${caseId}`);
    return res.data;
  },

  async verifyArchive(formData: FormData) {
    const res = await apiClient.post(`/archive/verify`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  getArchiveDownloadUrl(filename: string) {
    return `${API_BASE_URL}/archive/download/${filename}`;
  },

  getJobStreamUrl(jobId: string) {
    return `${API_BASE_URL}/jobs/${jobId}/stream`;
  },

  getSampleImageUrl(caseId: string, sampleId: string) {
    return `${API_BASE_URL}/cases/${caseId}/samples/${sampleId}/image`;
  },

  // Sprint 16: Admin Yönetim Paneli Metotları
  async getAdminUsers(params?: { role?: string; is_active?: boolean }) {
    const res = await apiClient.get<UserResponse[]>("/admin/users", { params });
    return res.data;
  },

  async createAdminUser(payload: { username: string; password: string; full_name: string; role: string }) {
    const res = await apiClient.post<UserResponse>("/admin/users", payload);
    return res.data;
  },

  async updateAdminUserRole(userId: string, role: string) {
    const res = await apiClient.put<UserResponse>(`/admin/users/${userId}/role`, { role });
    return res.data;
  },

  async updateAdminUserStatus(userId: string, isActive: boolean) {
    const res = await apiClient.put<UserResponse>(`/admin/users/${userId}/status`, { is_active: isActive });
    return res.data;
  },

  async resetAdminUserPassword(userId: string, newPassword: string) {
    const res = await apiClient.post<{ status: string; message: string }>(`/admin/users/${userId}/reset-password`, { new_password: newPassword });
    return res.data;
  },

  async getAdminAuditLogs(params?: { case_id?: string; examiner_id?: string; event_type?: string; limit?: number; offset?: number }) {
    const res = await apiClient.get<Array<{
      id: string;
      case_id?: string;
      event_type: string;
      action: string;
      examiner_id?: string;
      sha256_hash: string;
      details_json?: any;
      created_at_utc: string;
    }>>("/admin/audit-logs", { params });
    return res.data;
  },

  async refreshToken(refreshToken: string) {
    const res = await apiClient.post<{ access_token: string; refresh_token: string; token_type: string }>("/auth/refresh", { refresh_token: refreshToken });
    return res.data;
  },

  async logout(refreshToken?: string) {
    if (refreshToken) {
      try {
        await apiClient.post("/auth/logout", { refresh_token: refreshToken });
      } catch {
        // Sessiz hata yakalama
      }
    }
  },
};
