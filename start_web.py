"""
start_web.py — Launch the InventoryPro FastAPI web dashboard
=============================================================
Usage:
    python start_web.py
Opens http://localhost:8000 in the default browser automatically.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_deps():
    missing = []
    for pkg in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.0"]
        )


def _check_pipeline():
    required = [
        ROOT / "data" / "processed" / "baseline_summary.json",
        ROOT / "data" / "processed" / "optimization_summary.json",
        ROOT / "data" / "processed" / "backtest_summary.json",
    ]
    if not all(p.exists() for p in required):
        print("First-run: generating data pipeline outputs (~20 s)…")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("Pipeline failed:\n", result.stderr or result.stdout)
            sys.exit(1)
        print("Pipeline complete.")


def main():
    _check_deps()
    _check_pipeline()

    url = "http://localhost:8000"
    print(f"\n  InventoryPro Web Dashboard")
    print(f"  -> {url}\n")

    # Open browser after a short delay so the server has time to start
    def _open():
        time.sleep(1.5)
        webbrowser.open(url)

    import threading
    threading.Thread(target=_open, daemon=True).start()

    subprocess.run(
        [sys.executable, "-m", "uvicorn", "dashboard.api:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
