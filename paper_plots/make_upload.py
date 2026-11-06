"""Collect the rendered figures and tables into one folder laid out like the Overleaf project.

Rebuilds ``<PAPER_ARTIFACTS>/overleaf_upload`` from scratch, so nothing stale survives, and
fails if a file the paper includes is missing. Run it after every plot script::

    uv run python -m paper_plots.make_upload
"""

from __future__ import annotations

import shutil

from paper_plots.paths import PAPER_ARTIFACTS

OUT_DIR = PAPER_ARTIFACTS / "overleaf_upload"
# Paper folder -> rendered files, relative to PAPER_ARTIFACTS.
FILES: dict[str, tuple[str, ...]] = {
    "figures/appendix/sp": tuple(
        f"{ds}_methods/{name}_{ds}.pdf" for ds in ("imagenet", "cifar10") for name in ("bar_sp", "sp_arc")
    ),
    "figures/appendix/ood": tuple(
        f"{ds}_methods/bar_ood_{band}_{ds}.pdf" for ds in ("imagenet", "cifar10") for band in ("near", "far")
    ),
    "figures/appendix/ood/hist": (
        "imagenet_methods/ood_hist/ood_hist_imagenet_het_net_inaturalist.pdf",
        "imagenet_methods/ood_hist/ood_hist_imagenet_bayesian_inaturalist.pdf",
        "imagenet_methods/ood_hist/ood_hist_imagenet_credal_bnn_inaturalist.pdf",
        "cifar10_methods/ood_hist/ood_hist_cifar10_base_places365.pdf",
        "cifar10_methods/ood_hist/ood_hist_cifar10_credal_relative_likelihood_places365.pdf",
        "cifar10_methods/ood_hist/ood_hist_cifar10_sngp_places365.pdf",
    ),
    "figures/appendix/cov_eff": ("cifar10_methods/coverage_efficiency_cifar10.pdf",),
    "figures/appendix/al": (
        *(f"al_methods/bar_al_eu_tu_{ds}.pdf" for ds in ("openml_6", "openml_156", "cifar10")),
        *(
            f"al_methods/al_curves_one_col_{ds}_{notion}.pdf"
            for ds in ("openml_6", "openml_156")
            for notion in ("total", "epistemic", "aleatoric")
        ),
        "al_methods/al_curves_one_col_cifar10_total.pdf",
        "al_methods/al_curves_one_col_cifar10_epistemic.pdf",
    ),
    "tables": (
        "paper_tables/paper_table_imagenet.tex",
        "paper_tables/paper_table_cifar10.tex",
        "paper_tables/paper_table_al.tex",
    ),
}


def main() -> None:
    """Rebuild the upload folder and report how many files it holds."""
    missing = [rel for files in FILES.values() for rel in files if not (PAPER_ARTIFACTS / rel).exists()]
    if missing:
        msg = f"not rendered yet: {missing}"
        raise SystemExit(msg)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    for folder, files in FILES.items():
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)
        for rel in files:
            shutil.copy2(PAPER_ARTIFACTS / rel, OUT_DIR / folder)
    print(f"wrote {sum(len(files) for files in FILES.values())} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
