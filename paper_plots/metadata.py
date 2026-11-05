"""Shared display-name mappings.

Filesystem keys (cache paths, output filenames) keep using the lowercase
config strings the runs were logged with. Pretty strings are reserved for
titles, axis labels, and table headers.
"""

from __future__ import annotations

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cifar10h": "CIFAR-10H",
    "imagenet": "ImageNet",
    "imagenet1k": "ImageNet-1K",
    "imagenet200": "ImageNet-200",
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "svhn": "SVHN",
    "places365": "Places365",
    "textures": "Textures",
    "ssb_hard": "SSB-Hard",
    "ninco": "NINCO",
    "inaturalist": "iNaturalist",
    "openimage_o": "OpenImage-O",
    "stl10": "STL-10",
    "eurosat": "EuroSAT",
    "sun397": "SUN397",
    "tin": "TIN",
    "openml_6": "OpenML 6",  # adjust to the task's real name if you want
    "openml_156": "OpenML 156",  # adjust to the task's real name if you want
}


def dataset_display(name: str) -> str:
    """Return the canonical display string for a dataset key.

    Falls back to the key unchanged when no override is registered.
    """
    return DATASET_DISPLAY_NAMES.get(name, name)


# Labels mirror the ``label:`` field in src/probly_benchmark/configs/method/<name>.yaml.
# Entries marked INVENTED have no upstream label and were chosen here -- adjust if
# you want a different display string in bars / tables / scatter / histograms.
METHOD_LABELS: dict[str, str] = {
    "base": "Base",
    "batchensemble": "BatchEnsemble",
    "bayesian": "BNN",
    "credal_bnn": "CreBNN",
    "credal_ensembling": "CreEns",
    "credal_net": "CreNet",
    "credal_relative_likelihood": "CreRL",
    "credal_wrapper": "CreWra",
    "dare": "DARE",
    "ddu": "DDU",
    "deup": "DEUP",
    "dropconnect": "DropConnect",
    "dropout": "Dropout",
    "duq": "DUQ",
    "efficient_credal_prediction": "EffCre",
    "ensemble": "Deep Ensemble",
    "evidential_classification": "EDL",
    "het_net": "HetNet",
    "laplace": "Laplace",  # INVENTED: no label: field in laplace.yaml
    "natural_posterior_network": "NatPN",
    "posterior_network": "PostNet",
    "prior_network": "PriorNet",
    "sngp": "SNGP",
    "subensemble": "SubEnsemble",
    "vbll": "VBLL",
    # INVENTED -- synthetic calibration variants (see paper_plots.fetch_calibration);
    # no upstream method config exists, the label is whatever reads best in figures.
    "base_label_smoothing": "Label Smooth.",
    "base_label_relaxation": "Label Relax.",
    "base_vector_scaling": "Vector Scaling",
    "base_temperature_scaling": "Temp. Scaling",
    "base_conformal_tv": "CCP",
}


# Method keys that represent ``base`` with a non-default loss / calibration.
# In the SP / OOD bar plots these get the lightest of three blue tiers (see
# :func:`paper_plots.style.bar_color_for`).
BASE_VARIANT_KEYS: frozenset[str] = frozenset(
    {
        "base_label_smoothing",
        "base_label_relaxation",
        "base_temperature_scaling",
        "base_vector_scaling",
    },
)

# Credal-set method keys. SP / OOD bar plots tint these with the medium-light
# blue tier (see :func:`paper_plots.style.bar_color_for`).
CREDAL_METHOD_KEYS: frozenset[str] = frozenset(
    {
        "credal_wrapper",
        "credal_ensembling",
        "credal_bnn",
        "credal_net",
        "credal_relative_likelihood",
        "efficient_credal_prediction",
    },
)

# Methods that output an uncertainty score or a single distribution instead of a
# second-order distribution. Every method outside this set, the credal set, and the
# ``base`` variants counts as "2nd-order Distr." in the plots. HetNet is here because
# it models label noise only; its logit samples say nothing about the model.
DIRECT_METHOD_KEYS: frozenset[str] = frozenset(
    {
        "ddu",
        "deup",
        "duq",
        "het_net",
    },
)

# Methods that should never appear in any plot or table, even if data exists
# in the cache. Single source of truth -- aggregation / listing helpers in
# ``aggregate``, ``al_aggregate``, ``sp_arc``, ``ood_hist``, and
# ``coverage_efficiency`` all skip these.
EXCLUDED_METHODS: frozenset[str] = frozenset(
    {
        "credal_net",
    },
)


def method_label(name: str) -> str:
    """Return the canonical display string for a method key.

    Falls back to the key unchanged when no override is registered.
    """
    return METHOD_LABELS.get(name, name)
