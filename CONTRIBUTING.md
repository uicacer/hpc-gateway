# Contributing to hpc-as-api

Thank you for your interest in contributing. This document describes how to set up a development environment, run the test suite, and submit changes.

## Development setup

```bash
git clone https://github.com/uicacer/hpc-as-api
cd hpc-as-api
uv sync --extra dev        # installs all dev dependencies
uv run pytest -v           # should show 28 tests passing
```

If you don't have `uv`, install it with `pip install uv` or see [uv docs](https://docs.astral.sh/uv/).

## Running tests

```bash
# All tests (no Globus credentials needed — Globus SDK is mocked)
uv run --extra dev pytest -v

# A single test module
uv run --extra dev pytest tests/test_utils.py -v
```

The `test_compute.py` and `test_app.py` tests mock the Globus SDK so they run offline without any HPC credentials.

## Project structure

```
hpc_as_api/
  auth.py      — Globus token + API key authentication, rate limiting
  compute.py   — GlobusComputeClient: job submission and streaming
  crypto.py    — AES-256-GCM end-to-end encryption
  utils.py     — OpenAI multimodal message utilities
  app.py       — FastAPI routes (standalone + embeddable router)
paper/
  paper.md     — JOSS paper
  paper.bib    — BibTeX references
tests/
  test_utils.py    — message utility tests
  test_compute.py  — GlobusComputeClient unit tests
  test_app.py      — FastAPI route tests
docs/
  deployment.md    — production deployment guide
```

## Making changes

1. Fork the repository and create a branch: `git checkout -b feature/my-change`
2. Make your changes, add tests where appropriate
3. Run `uv run --extra dev pytest` to confirm tests pass
4. Run `uv run --extra dev ruff check hpc_as_api/` for style checks
5. Open a pull request with a clear description of what changed and why

## Reporting bugs / requesting features

Open an issue at <https://github.com/uicacer/hpc-as-api/issues>. Include:
- Python version (`python --version`)
- hpc-as-api version (`pip show hpc-as-api`)
- A minimal reproducing example
- The full error traceback

## Support and governance

This project is maintained by the [ACER group at UIC](https://acer.uic.edu). Questions and bug reports via GitHub Issues are the primary support channel. Response time is typically within one week.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
