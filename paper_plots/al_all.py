"""Run the full AL pipeline: bulk download, then classify + dedup.

``fetch_al`` is the slow network-bound stage; ``analyze_al`` is a pure-local
pass that writes the merged sink. Re-running this script after ``fetch_al`` is
incremental (raw files already on disk are skipped).

Run::

    uv run python -m paper_plots.al_all
"""

from __future__ import annotations

from paper_plots import analyze_al, fetch_al


def main() -> None:
    print("=== fetch_al ===")
    fetch_al.main()
    print("\n=== analyze_al ===")
    analyze_al.main()
    print("\nall done.")


if __name__ == "__main__":
    main()
