# AGENTS.md
## Directives for Agents:
- If you are creating new files add them to git via `git add <file>`. If you forget to do this, your changes will not be committed and may be lost. Do not add bloat files like `__pycache__` or `*.pyc` files or auto-generated files to git.
- Always run pre-commit checks with `uv run prek run --all-files` before committing to ensure that your code adheres to the project's style and quality standards.
- When adding public-facing features add docstrings in Google-Style format and American english.
## Useful Commands:
### Build Docs (only use these commands verbatim from the project root)
Incremental build (only changed examples and pages re-run):
```bash
uv run sphinx-build -j auto -b html docs/source docs/build/html
```
If an incremental build warns about API pages for objects that no longer exist (e.g. after switching branches), run `uv run python docs/source/_prune_stale_api.py` and rebuild.
Full build (re-run all examples):
```bash
rm -rf docs/source/api docs/source/auto_examples docs/source/gen_modules docs/build && FORCE_CLEAN=1 uv run sphinx-build -j auto -b html docs/source docs/build/html
```
Full build with errors on warning (to see if CI passes):
```bash
rm -rf docs/source/api docs/source/auto_examples docs/source/gen_modules docs/build && FORCE_CLEAN=1 uv run sphinx-build -j auto -b html docs/source docs/build/html -W
```
### Docs Versioning
The deployed site has two versions: `/stable/` (built from the newest release tag by `release-docs.yml`, stored as a `probly-docs-html.tar.gz` asset on that release) and `/latest/` (built from main on every push by `deploy-docs.yml`). Local builds always produce the "latest" flavor; set `DOCS_SOURCE_REF=<tag>` to preview the release flavor.
### Run Pre-commit (takes only 1s)
```bash
uv run prek run --all-files
```
### Only Run type checking
```bash
ty check <path-to-file-or-directory>
```
## Example files to look at:
Examples of how to use probly lives in the examples directory. You can find tutorials on how to use the pytraverser flexdispatch and so forth.
## How Dispatch Works (short version):
Probly uses three dispatch layers to stay backend-agnostic:
1. **`flexdispatch`** (from `flextype`) -- type-based function dispatch, like `functools.singledispatch` but more flexible. Core operations (`predict`, `calibrate`, `ensemble_generator`) use this to route to the right backend implementation based on the predictor's type.
2. **`flexdispatch_traverser`** (from `pytraverse`) -- walks a neural network's layer tree and dispatches per-layer transformations by type. Methods like `dropout` and `batchensemble` use a traverser to apply backend-specific transforms to each layer.
3. **`switchdispatch`** (`probly/utils/switchdispatch.py`) -- dispatches on value equality instead of type. Used for the predictor registry to map name strings to predictor classes.
**Typical flow** (e.g. `dropout(torch_model)`):
```
user calls dropout(model)
  -> @predictor_transformation validates & infers backend
  -> traverse(model, dropout_traverser)
     -> traverser visits each layer, dispatches by type
     -> torch handler (method/dropout/torch.py) wraps the layer
```
**Lazy backend loading**: backends register via `delayed_register` in `__init__.py` files using fully-qualified type strings from `probly/lazy_types.py` (e.g. `TORCH_MODULE = "torch.nn.modules.module.Module"`). This means torch/flax/sklearn are only imported when actually needed.
## Common Mistakes to do right:
- `ty` may still fail to treat `np.ndarray` as a structural subtype of our `ArrayLike` protocol even after relaxing method requirements. Keep bounds as `ArrayLike | np.ndarray` where needed and use local `cast("Any", ...)` when dispatching ndarray-specific dunder methods.
- Do not use special unicode characters where it is not necessary (comments, docstrings, variable names)
- Tests are split by backend. Put backend-agnostic checks in `test_common.py`, and backend-specific checks in files like `test_array.py`, `test_torch.py`, or `test_jax.py`. In backend-specific test files, call `pytest.importorskip("<backend>")` at the top and avoid per-test skip decorators for missing optional deps.
- Pickle default-state behavior is subtle: `object.__getstate__()` may return `None` even when an instance has a populated `__dict__`. Do not use `super().__getstate__()` as a drop-in replacement for pickle's default state extraction when implementing cooperative `__getstate__` wrappers.
