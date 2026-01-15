.. _contributing:

Contributing Guide
==================

``Probly`` is an open-source project, and we welcome contributions from the
community. Whether you maintain or expand the library, make a bug report,
or write a new documentation page your help is greatly appreciated.
``Probly`` is hosted on GitHub at https://github.com/pwhofman/probly.

The Installation and Quick Start Guide can be found here: :ref:`installation`.

   .. admonition:: Code of Conduct

      We expect all contributors to adhere to the
      `Code of Conduct <https://policies.python.org/python.org/code-of-conduct/>`_ :cite:`PSF2018CodeOfConduct`.
      By participating in this project, you agree to abide by its terms.

Here are some ways you can contribute:
Look for issues on the GitHub repository. These `issues <https://github.com/pwhofman/probly/issues>`_
are a great way to get started with contributing to ``probly``.

Submitting a bug report or a feature request
------------------------------------------------

Please adhere to the following guidelines when submitting a bug report or feature request:
1. Search the existing issues to see if your bug or feature request has already been reported.
2. Provide a clear and concise description of the bug or feature request.
3. Include steps to reproduce the bug, if applicable.
4. Provide any relevant information, such as your operating system, Python version, and ``probly`` version.
This information can be found by running:

   .. code-block:: sh

      python -c "import probly; print(probly.__version__)"



Contributing Code
----------------------
If you found an issue that you would like to work on, please fork the
repository and create a new branch for your changes. Once you have made
your changes, submit a pull request to the main repository. Please make
sure to include tests for your changes and update the documentation if necessary.

How to contribute
^^^^^^^^^^^^^^^^^^^^^^^^
1. Create an account on GitHub if you don't have one already.

2. Fork the `repository <https://github.com/pwhofman/probly>`_ on GitHub.
The "Fork" button is located at the top right corner of the repository page.
This creates a copy of the repository under your GitHub account.
You can also look up this `guide <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_
for more information on forking a repository.


3. Clone your forked ``probly`` repository to your local machine:

   .. code-block:: sh

      git clone
      cd probly

4. Create a new branch for your changes:

   .. code-block:: sh

      git checkout -b my-feature-branch

5. Make your changes to the codebase. Ensure that your code adheres to the project's coding
standards and includes appropriate tests.

6. Commit your changes with a descriptive commit message:

   .. code-block:: sh

      git add .
      git commit -m "Add feature X"

7. Push your changes to your forked repository:

   .. code-block:: sh

      git push origin my-feature-branch

8. Go to the original ``probly`` repository on GitHub and click on the
   "New pull request" button. Select your branch from your forked repository and submit the pull request.
   Provide a clear description of the changes you have made and why they are necessary.

Set up your development environment
------------------------------------
After you forked the repository, you can now set up your development environment for contributing to ``probly``:

1. Ensure you have Python 3.10 or higher installed on your machine.

2. `uv <https://docs.astral.sh/uv/>`_ :cite:`uv` can be used to manage the development environment.

   Run

   .. code-block:: sh

      uv sync --dev

   This will install all the required dependencies in your Python environment.

Add new examples
------------------------
We welcome new examples that demonstrate the capabilities of ``probly``.
If you have an idea for a new example, please discuss it with the maintainers first.
Once you have the ok from the maintainers, please create a new file in the
`examples/` directory and follow the existing structure and style of the examples.
You then can run the examples locally by following these steps:

1. Install the required dependencies:

   .. code-block:: sh

      pip install -r examples/requirements.txt

2. Navigate to the `examples/` directory:

   .. code-block:: sh

      cd examples

3. Run the example script using Python:

   .. code-block:: sh

      python example_script.py

4. Make sure to test your example and ensure that it runs correctly before submitting a pull request


Contributing to the Documentation
------------------------------------
We welcome contributions to the documentation. The documentation is located in the
`docs/` directory of the repository. If you find a typo, unclear explanation,
or missing information, please feel free to submit a pull request with your suggested changes.
If you want to add new sections or pages to the documentation, please discuss
your ideas with the maintainers first. If you get the ok from the maintainers,
please follow the existing structure and style of the documentation.
You can build the documentation locally by following these steps:

1. Install the required dependencies:

   .. code-block:: sh

      pip install -r docs/requirements.txt

2. Navigate to the `docs/` directory:

   .. code-block:: sh

      cd docs

3. Build the documentation using Sphinx:

   .. code-block:: sh

      make html

4. Open the generated HTML files located in the `docs/_build/html/` directory
in your web browser to review your changes.
5. Make sure to test your changes and ensure that the documentation builds
correctly before submitting a pull request.


Documentation test warnings should be fixed by running:

   .. code-block:: sh

      sphinx-build -b linkcheck . _build/linkcheck


Doctrings and style guide
--------------------------------
When contributing code, please ensure that your code follows the
`Google Style Guide <https://google.github.io/styleguide/pyguide.html#docstrings>`_ :cite:`GooglePythonStyleGuide`
style guide and includes appropriate docstrings.
For linting and formatting, ``Probly`` uses ``Ruffly``. The rules can be found in the `pyproject.toml <https://github.com/pwhofman/probly/blob/main/pyproject.toml>`_ file. If you set up your develelopment environment
correctly, the pre-commit hooks will automatically run Ruffly on your code before each commit.


CI checks
----------------
All contributions are subject to continuous integration (CI) checks to ensure
code quality and consistency. CI checks help identify issues such as failing tests,
code style violations, and other potential problems.
When you submit a pull request, the CI checks will automatically run on your code.
You should ensure that all CI checks pass before your pull request can be merged.

The following CI checks are performed during a pull request:

- Code quality checks
      `pre-commit` is used to run various code quality checks on the codebase.
      This ist run throuch `uv`.

- Install and import check
   see the :ref:`installation` section for more details.

- Unit tests
      `pytest` is used to run the unit tests for the project. This is done
      across four different environments: ubuntu-latest (all_ml),
      ubuntu-latest(torch), windows-latest and macos-latest with Python 3.11.

- Documentation build checks
      Sphinx is used to build the documentation and check for any build errors.
      Install all dependencies listed in `docs/requirements.txt` to ensure the
      documentation builds correctly.
      You can build the documentation locally by running:
      .. code-block:: sh

         sphinx-build -b html docs/ docs/_build/html
- Test coverage
      `coverage.py` is used to measure the test coverage of the codebase.
      The goal is to maintain a high level of test coverage to ensure that
      the code is well-tested and reliable.
      This runs after the unit tests, code quality checks and documentation
      build checks have passed. It generates a coverage report `coverage.xml`
      that shows the percentage of code that is covered by tests.


The following CI checks are performed specifically for documentation contributions:

- Link checks
      Sphinx linkcheck builder is used to check for broken links in the documentation.

- Type checking
     `mypy` is used to check for type hints and type consistency in the codebase.

- Formatting checks
     `Ruffly` is run to ensure consistent code formatting across the codebase.

- Example checks
      The examples in the `examples/` directory are run to ensure they work correctly.

- Doctests
      Doctests in the documentation are run to ensure they work correctly.
      If they fail, artifacts are generated to help debug the issues.

- You can view the status of the CI checks on the pull request page.
      A checkmark indicates that the check has passed, while a red cross
      indicates that the check has failed. A spinning circle indicates that
      the check is still in progress.
      Click on any jobs to view detailed logs and information about the check.

You can also run the CI checks locally before submitting your pull request.
This can help you identify and fix issues before they are caught by the CI system.

   .. code-block:: sh

      # Build the documentation
      sphinx-build -b html docs/ docs/_build/html

      # Run unit tests
      pytest

      # Run link checks
      sphinx-build -b linkcheck docs/ docs/_build/linkcheck

      #Run doctests
      sphinx-build -b doctest docs/ docs/_build/doctest

      # Run mypy for type checking
      mypy probly/

      # Run ruffly for formatting checks
      ruff check probly/

      # Run examples
      python examples/example_script.py


Automated Contributions Policy
--------------------------------
Please do not submit automated contributions (e.g., generated by AI tools)
without prior approval from the maintainers. We value genuine contributions
that reflect a clear understanding of the project and its goals.
