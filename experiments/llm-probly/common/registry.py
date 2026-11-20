"""Method registry (E2): one table that says what every uncertainty method is and what it costs.

The runner, the summary table and the selector (D1) read this table instead of
the plugin modules, so the metadata of a method lives in one place. A plugin
file in ``methods/`` only needs ``NAME`` and ``score()`` (see CONTRACT.md).

Fields of a method:

- ``family``: ``agreement`` (consistency of samples), ``self_eval`` (the model
  judges its own answer), ``baseline`` (read from the greedy answer) or
  ``hybrid`` (confidence x consistency).
- ``access``: what the method needs from the model.

  - ``text_only``: generated text only, works for any model behind an API.
  - ``logprobs``: token log-probabilities of generated or prompted text, as
    APIs with a ``logprobs`` option return them.
  - ``white_box``: the full next-token distribution or the weights, so only a
    local model.

- ``cost``: relative cost per question, generation included. The unit is one
  LLM generation pass, so ``1`` is the greedy answer alone and a method that
  also reads the 10 cached samples costs ``11``.
- ``needs_model``: the runner loads the LLM and passes it to ``score()``.
- ``uses_nli``: the method also runs the NLI model on sample pairs (cheap
  compared with the LLM, but it is reported separately).

Example::

    from common.registry import REGISTRY, candidates

    REGISTRY["semantic_entropy"].access  # "logprobs"
    candidates(access="text_only", max_cost=11)  # what an API-only user can afford
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

Family = Literal["agreement", "self_eval", "baseline", "hybrid"]
Access = Literal["text_only", "logprobs", "white_box"]

FAMILIES: tuple[str, ...] = get_args(Family)
# From least to most access: a method with access level i also runs where level j > i is available.
ACCESS_LEVELS: tuple[str, ...] = get_args(Access)

GREEDY = 1  # one generation pass for the greedy answer
SAMPLES = 10  # cached samples per question (scripts/generate.py)

METHODS_DIR = Path(__file__).resolve().parent.parent / "methods"


@dataclass(frozen=True)
class MethodSpec:
    """Metadata of one uncertainty method.

    Attributes:
        name: Same as the plugin's ``NAME`` and the result file name.
        family: Method family, one of ``FAMILIES``.
        access: What the method needs from the model, one of ``ACCESS_LEVELS``.
        cost: LLM generation passes per question, greedy answer included.
        needs_model: True if ``score()`` needs the loaded LLM.
        uses_nli: True if the method also runs the NLI model.
        probability: True if ``1 - uncertainty`` is a probability of being correct
            (ECE is computed only for these).
        note: One line on where the cost comes from.
    """

    name: str
    family: Family
    access: Access
    cost: float
    needs_model: bool = False
    uses_nli: bool = False
    probability: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Check the field values, so a typo fails when the module is imported."""
        if self.family not in FAMILIES:
            msg = f"{self.name}: family must be one of {FAMILIES}, got {self.family!r}"
            raise ValueError(msg)
        if self.access not in ACCESS_LEVELS:
            msg = f"{self.name}: access must be one of {ACCESS_LEVELS}, got {self.access!r}"
            raise ValueError(msg)
        if not self.cost >= 0:
            msg = f"{self.name}: cost must be a number >= 0, got {self.cost}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Fields as a plain dict (for JSON, the profile and the report)."""
        return asdict(self)


_SPECS = (
    # --- baselines: read the cached greedy answer only ---------------------------------------
    MethodSpec("msp", "baseline", "logprobs", GREEDY, note="sum of greedy token log-probs"),
    # Perplexity = exp(-likelihood_mean) ranks the answers identically, so it is not a separate method.
    MethodSpec(
        "likelihood_mean", "baseline", "logprobs", GREEDY, note="mean greedy token log-prob; = perplexity ranking"
    ),
    MethodSpec(
        "mean_token_entropy",
        "baseline",
        "white_box",
        GREEDY,
        note="needs the full next-token distribution at every step",
    ),
    MethodSpec("dummy_random", "baseline", "text_only", GREEDY, note="random reference, no signal"),
    # --- agreement: compare the 10 cached samples ----------------------------------------------
    MethodSpec("lexical_similarity", "agreement", "text_only", GREEDY + SAMPLES, note="Rouge-L between samples"),
    MethodSpec(
        "discrete_semantic_entropy",
        "agreement",
        "text_only",
        GREEDY + SAMPLES,
        uses_nli=True,
        note="cluster sizes from NLI, no likelihoods",
    ),
    MethodSpec(
        "semantic_entropy",
        "agreement",
        "logprobs",
        GREEDY + SAMPLES,
        uses_nli=True,
        note="NLI clusters weighted by sample likelihoods",
    ),
    # --- self-evaluation: ask the model about its own answer -----------------------------------
    MethodSpec(
        "p_true_zeroshot",
        "self_eval",
        "logprobs",
        GREEDY + 1,
        needs_model=True,
        probability=True,
        note="one scoring pass for P(A) vs P(B)",
    ),
    MethodSpec(
        "p_true_brainstorm",
        "self_eval",
        "logprobs",
        GREEDY + SAMPLES + 1,
        needs_model=True,
        probability=True,
        note="the cached samples go into the prompt, then one scoring pass",
    ),
    MethodSpec(
        "p_true_empirical",
        "self_eval",
        "text_only",
        GREEDY + SAMPLES,
        needs_model=True,
        probability=True,
        note="10 sampled True/False judgments",
    ),
    MethodSpec(
        "verbalized_2s",
        "self_eval",
        "text_only",
        GREEDY + 1,
        needs_model=True,
        probability=True,
        note="one short generation for the stated confidence",
    ),
)

REGISTRY: dict[str, MethodSpec] = {spec.name: spec for spec in _SPECS}


def get(name: str) -> MethodSpec:
    """Return the spec of one method.

    Args:
        name: Method name, as in the result file.

    Returns:
        The method's spec.

    Raises:
        KeyError: If the method is not registered; the message lists the known names.
    """
    if name not in REGISTRY:
        msg = f"method {name!r} is not in common/registry.py; known: {sorted(REGISTRY)}"
        raise KeyError(msg)
    return REGISTRY[name]


def family_of(name: str) -> str:
    """Family of a method, or ``""`` for result files of unregistered (e.g. week-1) methods."""
    spec = REGISTRY.get(name)
    return spec.family if spec else ""


def candidates(
    access: str = "white_box",
    max_cost: float = float("inf"),
    family: str | None = None,
) -> list[MethodSpec]:
    """Methods that can run with the given model access and cost budget, cheapest first.

    Args:
        access: Access to the model. ``white_box`` allows every method,
            ``logprobs`` allows ``logprobs`` and ``text_only`` methods, and
            ``text_only`` only ``text_only`` methods.
        max_cost: Highest allowed cost per question (generation passes).
        family: Only methods of this family, if given.

    Returns:
        The matching specs, sorted by cost and then by name.

    Raises:
        ValueError: If ``access`` is not one of ``ACCESS_LEVELS``.
    """
    if access not in ACCESS_LEVELS:
        msg = f"access must be one of {ACCESS_LEVELS}, got {access!r}"
        raise ValueError(msg)
    allowed = ACCESS_LEVELS[: ACCESS_LEVELS.index(access) + 1]
    found = [
        spec
        for spec in REGISTRY.values()
        if spec.access in allowed and spec.cost <= max_cost and (family is None or spec.family == family)
    ]
    return sorted(found, key=lambda spec: (spec.cost, spec.name))


def plugin_names() -> list[str]:
    """File names (without ``.py``) of all plugins in ``methods/``, helpers excluded."""
    return sorted(path.stem for path in METHODS_DIR.glob("*.py") if not path.name.startswith("_"))


def load_plugin(name: str) -> ModuleType:
    """Import the plugin module of a registered method.

    Args:
        name: Method name; the plugin file is ``methods/<name>.py``.

    Returns:
        The imported module, which has ``NAME`` and ``score``.

    Raises:
        KeyError: If the method is not registered.
        AttributeError: If the plugin lacks ``NAME`` or ``score``, or its ``NAME`` differs.
    """
    get(name)
    module = importlib.import_module(f"methods.{name}")
    for attr in ("NAME", "score"):
        if not hasattr(module, attr):
            msg = f"methods/{name}.py has no {attr} (see CONTRACT.md)"
            raise AttributeError(msg)
    if name != module.NAME:
        msg = f"methods/{name}.py has NAME = {module.NAME!r}; it must equal the file name"
        raise AttributeError(msg)
    return module


def score_function(name: str) -> Callable[..., dict[str, Any]]:
    """The ``score()`` function of a registered method."""
    return load_plugin(name).score


def check_registry() -> list[str]:
    """Compare the registry with the plugin files and return every mismatch.

    A plugin without an entry, an entry without a plugin, and a plugin whose
    old ``FAMILY`` / ``NEEDS_MODEL`` constants disagree with its entry are all
    reported. An empty list means the two agree.
    """
    problems = []
    files = set(plugin_names())
    problems += [f"methods/{n}.py has no entry in common/registry.py" for n in sorted(files - set(REGISTRY))]
    problems += [f"registry entry {n!r} has no methods/{n}.py" for n in sorted(set(REGISTRY) - files)]
    for name in sorted(files & set(REGISTRY)):
        module = load_plugin(name)
        spec = REGISTRY[name]
        if hasattr(module, "FAMILY") and spec.family != module.FAMILY:
            problems.append(f"methods/{name}.py: FAMILY {module.FAMILY!r} != registry {spec.family!r}")
        if hasattr(module, "NEEDS_MODEL") and spec.needs_model != module.NEEDS_MODEL:
            problems.append(f"methods/{name}.py: NEEDS_MODEL {module.NEEDS_MODEL} != registry {spec.needs_model}")
    return problems
