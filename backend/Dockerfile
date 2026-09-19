# FHDSS Backend - Kurumsal Adli Karar Destek Sistemi Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Sistem bağımlılıkları (OpenCV, Pango, Cairo, Adli Belge Kütüphaneleri)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Kaynak kodları ve adli modülleri kopyala
COPY backend ./backend
COPY cv ./cv
COPY ml ./ml

# Çıktı ve veri dizinlerini hazırla
RUN mkdir -p /app/artifacts/samples /app/artifacts/reports /app/artifacts/archives

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
