"""
FHDSS (Adli Belge ve El Yazısı Karar Destek Sistemi) Tekil Başlatıcı (Single Terminal Runner).

Bu script hem Python FastAPI Backend'ini (Port 8000) hem de Next.js Frontend'ini (Port 3000)
tek bir terminalde eşzamanlı olarak başlatır, logları renklendirir ve çakışan portları otomatik temizler.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)


def clean_port_windows(port: int):
    """Windows üzerinde belirtilen portu kullanan eski süreçleri sonlandırarak EADDRINUSE hatasını önler."""
    try:
        cmd = f"netstat -ano | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        lines = output.strip().split("\n")
        killed_pids = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                if pid and pid != "0" and pid not in killed_pids and int(pid) != os.getpid():
                    subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(f"powershell -Command \"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue\"", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    killed_pids.add(pid)

        if killed_pids:
            print(f"🧹 Port {port} üzerindeki eski işlemler temizlendi (PID: {', '.join(killed_pids)}).")
    except Exception:
        pass


def stream_logs(process: subprocess.Popen, prefix: str, color_code: str):
    """Süreç çıktılarını eşzamanlı olarak renklendirip ana terminale basar."""
    reset_code = "\033[0m"
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"{color_code}[{prefix}]{reset_code} {line.rstrip()}")
    except Exception:
        pass


def main():
    print("\n" + "=" * 70)
    print(" 🚀 FHDSS - Adli Karar Destek Sistemi Başlatılıyor")
    print("=" * 70)

    # 1. Portları temizle (Eski asılı kalan süreçleri kapat)
    clean_port_windows(8000)
    clean_port_windows(3000)
    time.sleep(1)

    # 2. Backend Sürecini Başlat (FastAPI)
    backend_cmd = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
        "--reload-dir",
        str(ROOT_DIR / "backend"),
        "--reload-dir",
        str(ROOT_DIR / "cv"),
        "--reload-dir",
        str(ROOT_DIR / "ml"),
    ]

    print("\n🟢 [1/2] FastAPI Backend başlatılıyor (http://127.0.0.1:8000)...")
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    # 3. Frontend Sürecini Başlat (Next.js)
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]

    print("🟣 [2/2] Next.js Frontend başlatılıyor (http://localhost:3000)...")
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    print("\n" + "-" * 70)
    print(" ✨ TÜM SİSTEMLER AKTİF:")
    print("   👉 Arayüz (İş İstasyonu) : http://localhost:3000")
    print("   👉 Arka Uç API Dokümanı  : http://localhost:8000/docs")
    print("   👉 Kapatmak için         : Ctrl + C")
    print("-" * 70 + "\n")

    # Thread'ler ile logları yönlendir
    # Cyan: \033[96m, Magenta: \033[95m
    t_backend = threading.Thread(target=stream_logs, args=(backend_proc, "BACKEND ", "\033[96m"), daemon=True)
    t_frontend = threading.Thread(target=stream_logs, args=(frontend_proc, "FRONTEND", "\033[95m"), daemon=True)

    t_backend.start()
    t_frontend.start()

    try:
        while True:
            time.sleep(0.5)
            # Eğer süreçlerden biri beklenmedik şekilde çökerse kontrol et
            if backend_proc.poll() is not None:
                print("\n⚠️ Backend süreci durdu.")
                break
            if frontend_proc.poll() is not None:
                print("\n⚠️ Frontend süreci durdu.")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 FHDSS sistemleri güvenli şekilde kapatılıyor...")
    finally:
        # Alt süreçleri nazikçe ve tam olarak sonlandır
        try:
            if sys.platform == "win32":
                subprocess.run(f"taskkill /F /T /PID {backend_proc.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(f"taskkill /F /T /PID {frontend_proc.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                backend_proc.terminate()
                frontend_proc.terminate()
        except Exception:
            pass

        clean_port_windows(8000)
        clean_port_windows(3000)
        print("✅ Tüm sunucular kapatıldı ve portlar serbest bırakıldı.\n")


if __name__ == "__main__":
    main()
