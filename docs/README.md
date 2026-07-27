# DMK documentation

Sphinx + Doxygen/Breathe sources for the DMK documentation, published on Read the Docs.
This file is developer-facing and is intentionally *not* part of the Sphinx toctree, so it
does not appear on the published site.

## Building locally

Prerequisites:

- `doxygen` on your `PATH`. On FI resources:
  ```bash
  module load modules/2.4-20250724 doxygen
  ```
- A `python` with `sphinx`, `breathe`, and `sphinx_rtd_theme` installed:
  ```bash
  pip install -r requirements.txt   # breathe, sphinx_rtd_theme (sphinx assumed present)
  ```

Build:

```bash
cd docs
make html          # runs doxygen first, then sphinx-build; output in docs/_build/html/
make clean         # remove _build/ and the generated doxygen/ XML
```

`make html` invokes Sphinx as `python -m sphinx` so it uses the same interpreter as your
`python` (where `breathe`/`sphinx_rtd_theme` live). If that `python` is not the one you want,
override it:

```bash
make html PYTHON=/path/to/python
```

## Publishing on Read the Docs

### Version model

RTD builds one "version" per git ref:

- **`latest`** → the default branch (`main`): dev/bleeding-edge docs. Always present.
- **`stable`** → the highest semver **tag**: activates itself once a matching tag is pushed.
- **individual tags** (`v1.0.0`, `v1.1.0`, …): each browsable as its own version.

### One-time setup

1. Import the repo on <https://readthedocs.org> via the GitHub connection so it installs the
   **GitHub App** (needed for PR preview builds and status checks).
2. Set the **default version** to `stable` (Admin → Settings), so the bare docs URL shows the
   latest release while `latest` remains available for `main`.
3. Enable **"Build pull requests for this project"** (Admin → Settings). Each PR then gets an
   ephemeral, non-indexed preview build plus a check on the PR.
4. Add an **Automation Rule** (Admin → Automation Rules) to auto-activate release tags:
   *Activate version*, match `^v(\d+\.\d+\.\d+)$`, type **Tag**.

### Releasing

- Tag releases with semver, prefixed `v` (e.g. `v1.0.0`). RTD parses these to choose `stable`
  and to sort the version menu. The version string shown in the docs is derived from the tag
  (see the `READTHEDOCS_VERSION` handling in `conf.py`).
- **Gotcha:** tags are frozen snapshots, so `.readthedocs.yaml` and this `docs/` tree must
  already exist **at the tagged commit**. Merge documentation changes to `main` *before*
  cutting a tag; later docs fixes do not apply retroactively to existing tags.
