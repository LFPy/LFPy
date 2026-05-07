# GitHub Copilot Instructions for LFPy

## Project overview

LFPy is a Python package for calculating extracellular potentials and related measures from multicompartment neuron models. It is built on top of the [NEURON simulator](https://www.neuron.yale.edu/neuron) and [LFPykit](https://github.com/LFPy/LFPykit). Performance-critical routines are implemented as Cython extensions (`.pyx` files in `LFPy/`).

## Language and style

- Python 3.10+ only (`python = "^3.10"` in `pyproject.toml`).
- Follow [PEP 8](https://peps.python.org/pep-0008/) style. The CI runs `flake8` to enforce this.
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public classes and functions.
- Do not add inline comments unless they meaningfully explain non-obvious logic.
- Prefer explicit imports over wildcard imports.

## Packaging

- The project uses **Poetry** with `pyproject.toml` as the single source of truth for metadata, dependencies, and build configuration.
- A `build.py` script handles Cython extension compilation under the Poetry build system.
- Do **not** modify or re-introduce `setup.py` or `setup.cfg`.
- Minimum dependency versions are pinned in `pyproject.toml`; update them there (not in `requirements.txt`, which is kept only for legacy reference).
- Optional dependency groups: `tests` (pytest) and `docs` (sphinx, numpydoc, sphinx-rtd-theme, m2r2).

## Dependencies

| Package   | Minimum version |
|-----------|----------------|
| Python    | 3.10           |
| NEURON    | 9.0.1          |
| numpy     | 1.8            |
| scipy     | 0.14           |
| Cython    | 3.0            |
| h5py      | 2.5            |
| lfpykit   | 0.6.2          |

## Testing

- Tests live in `LFPy/test/`. Run them with:
  ```bash
  pytest LFPy/test/
  ```
- Install the test extra before running: `pip install -e ".[tests]"`.
- Do not remove or disable existing tests. New features should include matching tests.

## Cython extensions

- `.pyx` source files are in `LFPy/`. Compiled `.so`/`.pyd` files are build artefacts and must not be committed.
- Always build extensions before running tests that depend on them: `python build.py build_ext --inplace` or `pip install -e .`.

## CI (GitHub Actions)

- Workflows are in `.github/workflows/`.
- `python-app.yml`: tests Python 3.10–3.14 on Ubuntu; macOS runner for Python 3.14 only. Windows is excluded (`pip install neuron` is not supported on Windows).
- Use `actions/checkout@v4` and `actions/setup-python@v5` in all workflows.
- All shell steps must set `shell: bash`.

## Documentation

- Documentation source is in `doc/` and is built with Sphinx.
- ReadTheDocs configuration is in `.readthedocs.yaml` (pip-based, Python 3.12, `ubuntu-lts-latest`).
- Install the docs extra to build locally: `pip install -e ".[docs]"`, then `cd doc && make html`.
- The `m2r2` Sphinx extension is required (converts Markdown → reStructuredText). It is included in the `docs` extras.

## Security and licensing

- The project is licensed under GPL-3.0-or-later. Do not introduce dependencies with incompatible licenses.
- Do not commit secrets, credentials, or generated build artefacts.
