from fastapi import APIRouter

from backend.app.api.v1.endpoints import admin, analysis, archive, audit, auth, biometrics, cases, jobs

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Kimlik Doğrulama ve RBAC"])
api_router.include_router(admin.router, prefix="/admin", tags=["Sistem Yönetimi ve RBAC Denetimi"])
api_router.include_router(cases.router, prefix="/cases", tags=["Vaka ve Numune Yönetimi"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Adli Karar Destek ve Analiz"])
api_router.include_router(biometrics.router, prefix="/biometrics", tags=["Dinamik Biyometrik İmza İncelemesi"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Asenkron Analiz İşleri ve SSE"])
api_router.include_router(audit.router, prefix="/audit", tags=["Adli Denetim İzi ve Hash Zinciri"])
api_router.include_router(archive.router, prefix="/archive", tags=["Adli Arşiv Paketi (.ZIP) ve Doğrulama"])

