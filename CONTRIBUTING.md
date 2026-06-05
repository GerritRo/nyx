# Contributing to nyx

## Development Setup

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-username>/nyx.git
cd nyx
git remote add upstream https://github.com/GerritRo/nyx.git

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify everything works
pytest
```

---

## Workflow (Gitflow)

We use a **Gitflow** model: `main` holds tagged releases, `dev` is the
integration branch. All work happens on short-lived branches off `dev`.

| Branch prefix      | Purpose                 | Target  |
|--------------------|-------------------------|---------|
| `feature/<name>`   | New features            | `dev`   |
| `bugfix/<name>`    | Bug fixes               | `dev`   |
| `hotfix/<x.y.z>`   | Urgent fixes in prod    | `main`  |
| `release/<x.y.z>`  | Release prep            | `main`  |

**Typical contribution flow:**

```bash
git fetch upstream && git checkout -b feature/my-change upstream/dev

# ... develop, commit, test ...

git push origin feature/my-change
# Open a PR targeting dev
```

Keep branches short-lived. Rebase on `upstream/dev` before opening a PR.

---

## Committing with Commitizen

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce
[Conventional Commits](https://www.conventionalcommits.org/) and generate
changelogs automatically. Use `cz commit` instead of `git commit`:

```bash
cz commit
```

This walks you through an interactive prompt:

```
? Select the type of change you are committing: (Use arrow keys)
 » fix: A bug fix
   feat: A new feature
   docs: Documentation only changes
   refactor: A code change that neither fixes a bug nor adds a feature
   perf: A code change that improves performance
   test: Adding missing or correcting existing tests
   build: Changes that affect the build system or dependencies
   ci: Changes to CI configuration files and scripts
   chore: Other changes that don't modify src or test files

? What is the scope of this change? (press enter to skip)
  core, atmosphere, emitter, instrument, utils

? Write a short, imperative description of the change:
  > add lunar phase correction to Jones2013

? Provide additional contextual information (press enter to skip):
  > The phase angle was not correctly mapped for waning phases

? Is this a BREAKING CHANGE?  No
? Footer (press enter to skip, e.g. "Closes #42"):
  > Fixes #12
```

Result: `fix(emitter): add lunar phase correction to Jones2013`

You can also write commit messages manually -- the format is:

```
<type>(<scope>): <description>
```

Breaking changes use `!` after the type: `feat(core)!: require explicit bandpass`

**Why this matters:** `CHANGELOG.md` is generated directly from these commit
messages at release time via `cz bump --changelog`.

---

## Pull Requests

- **Title** follows Conventional Commits format (it becomes the merge commit message)
- **Target branch** is `dev` (unless it's a hotfix targeting `main`)
- **PR checklist:**
  - [ ] Tests pass (`pytest`)
  - [ ] Lint passes (`ruff check nyx`)
  - [ ] Types pass (`mypy nyx`)
  - [ ] New code has tests

---

## Code Quality

### Quick reference

```bash
ruff check nyx                          # Lint
ruff check --fix nyx                    # Lint + auto-fix
ruff format nyx                         # Format
mypy nyx                                # Type check
pytest                                  # Tests
pytest --cov=nyx --cov-report=html      # Coverage report
```

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
# Open docs/_build/html/index.html
```

- **Example notebooks** go in `docs/examples/` and must be added to `docs/examples/index.rst`

---

## Release Process

Maintainers only. Uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

1. Create `release/x.y.z` from `dev`
2. `cz bump --changelog` to bump version + generate changelog
3. Run full test suite + benchmarks + doc build
4. Merge into `main`, tag `vx.y.z`, backmerge into `dev`

---

## Getting Help

- [GitHub Issues](https://github.com/GerritRo/nyx/issues) for bugs and feature requests
- [Documentation](https://gerritro.github.io/nyx/)
- Email: gerrit.roellinghoff@fau.de

If unsure about a change, open an issue first to discuss the approach.
