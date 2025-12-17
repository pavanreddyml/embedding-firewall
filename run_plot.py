# file: scripts/plot_results.py
from __future__ import annotations

from pathlib import Path

from embfirewall.viz import write_all_figures

RUN_ID = ""

def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


# -----------------------------
# GLOBAL PATHS (edit this file)
# -----------------------------
IN_COLAB = _in_colab()

LOCAL_BASE_DIR = "."
COLAB_BASE_DIR = "/content/drive/MyDrive/research/embfirewall"  # <-- change to your folder on Drive

WORKING_DIR = LOCAL_BASE_DIR
STORAGE_DIR = COLAB_BASE_DIR if IN_COLAB else LOCAL_BASE_DIR

RUN_DIR = str(Path(STORAGE_DIR) / "runs" / RUN_ID)

RESULTS_PATH = str(Path(RUN_DIR) / "results.json")
FIGURES_DIR = str(Path(STORAGE_DIR) / "figures")
# -----------------------------


def main() -> None:
    print(f"[plot_results] IN_COLAB={IN_COLAB}")
    print(f"[plot_results] BASE_DIR={WORKING_DIR}")
    print(f"[plot_results] RUN_DIR={RUN_DIR}")
    print(f"[plot_results] RESULTS_PATH={RESULTS_PATH}")
    print(f"[plot_results] FIGURES_DIR={FIGURES_DIR}")

    rp = Path(RESULTS_PATH)
    if not rp.exists():
        raise SystemExit(f"[plot_results] Missing results.json: {RESULTS_PATH}")

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    write_all_figures(str(rp), FIGURES_DIR)
    print(f"[plot_results] wrote figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
