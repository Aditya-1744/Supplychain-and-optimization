"""
start.py  —  Single-command launcher for the Supply Chain dashboard
====================================================================
Solves problem #1: "I have to run backend and frontend manually."

Just run:
    python start.py

What it does:
  1. Checks that required Python packages are installed (streamlit, plotly).
     Offers to install them if missing.
  2. Runs the full analysis pipeline (scripts/run_pipeline.py) if the
     data/processed/ outputs don't exist yet.  Safe to re-run — the pipeline
     skips regeneration if outputs are already present.
  3. Launches `streamlit run dashboard/app.py` in the same process, so there
     is no separate "backend" to manage.  Streamlit IS the server; it imports
     src/ modules directly.

There is no separate backend process.  The Streamlit app calls
src/models/inference.py for live predictions, which in turn calls the
existing model classes — all in the same Python process.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── 1. Dependency check ───────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "streamlit": "streamlit>=1.33",
    "plotly":    "plotly>=5.20",
}

missing = []
for module, pip_spec in REQUIRED_PACKAGES.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(pip_spec)

if missing:
    print(f"\n  Missing packages: {', '.join(missing)}")
    answer = input("  Install them now? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("  Installed.\n")
    else:
        print("  Cannot start without required packages. Exiting.")
        sys.exit(1)

# ── 2. Pipeline (idempotent — skips if outputs already present) ───────────────
print("  Checking pipeline outputs...")
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--quiet"],
    capture_output=False,   # let pipeline print its own progress to the terminal
)
if result.returncode != 0:
    print("\n  Pipeline failed. Fix the error above, then re-run `python start.py`.")
    sys.exit(1)

# ── 3. Launch Streamlit (replaces this process — no extra terminal needed) ────
print("\n  Starting dashboard — open http://localhost:8501 in your browser.\n")
print("  Press Ctrl+C to stop.\n")
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(ROOT / "dashboard" / "app.py"),
    "--server.headless", "true",
])
