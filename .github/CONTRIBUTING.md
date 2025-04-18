# Contributing to probly 🏔️
Probly is still in early development and we welcome contributions
in many forms. If you have an idea for a new feature, a bug fix, or
any other suggestion for improvement, please open an issue on GitHub.
If you would like to contribute code, keep reading!

## What to work on ❓
We want to offer support for PyTorch, HuggingFace, and sklearn models. We are interested in
any contributions that translate existing features to these libraries. Furthermore, we are
interested in any new features within the following scope:
- Representation methods;
- Quantification methods;
- Calibration methods;
- Downstream tasks;
- Datasets and dataloaders.

If you have other suggestions, please open an issue on GitHub such that we can discuss it.

## Development setup 🔬
The recommended workflow for contributing to probly is:
1. Fork the `main` branch of repository on GitHub.
2. Clone the fork locally.
3. Commit the changes.
4. Push the changes to the fork.
5. Create a pull request to the `main` branch of the repository.

### Setting up the development environment
Once you have cloned the fork, you can set up your development environment.
You will need a Python 3.10+ environment. We recommend using [uv](https://docs.astral.sh/uv/) as a package manager.
To set up the development environment, run the following command:
```sh
uv sync --dev
```
This will install all the required dependencies in your Python environment.

## Guidelines 🚓
Here are some guidelines to follow when contributing to probly.

### General
If you use code from other sources, make sure to carefully look at the license and give credit to the original author(s).
If the feature you are implementing is based on a paper, make sure to include a reference
in the docstring.

### Code style
We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, the rules of which can be found in the [pyproject.toml](../pyproject.toml) file.
However, if your development environment is set up correctly, the Ruff pre-commit hook should take care of this for you.

### Documentation
If you are adding new features, make sure to document them in the docstring;
our docstrings follow the [Google Style Guide](https://google.github.io/styleguide/pyguide.html#docstrings).
