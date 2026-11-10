# Contributing to SmartPump AI

Welcome to the **SmartPump AI** project! This repository is built collaboratively by a team of three students for **Classwork 3.12**.

To maintain clean code quality, avoid merge conflicts, and ensure that our teacher can clearly track each member's contributions, all team members must follow this guide.

---

## Team Responsibilities Overview

| Student Name | Student ID | GitHub | Role | Primary Technical Domain | Key Files |
|:---|:---|:---|:---|:---|:---|
| **Biruk Ayitenew** | WOUR/0483/16 | [@Biruk1997](https://github.com/Biruk1997) | **Member 1** | Data Engineering & Exploration | `src/preprocessing.py`, `src/feature_engineering.py`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb` |
| **Abrham Gebremedhin** | WOUR/0082/16 | [@abrilohd](https://github.com/abrilohd) | **Member 2** | Machine Learning & Evaluation | `src/train.py`, `notebooks/03_*.ipynb`, `results/` |
| **Israel Abera** | WOUR/4167/16 | [@Isru96](https://github.com/Isru96) | **Member 3** | Integration, Documentation & QA | `README.md`, `docs/`, `tests/`, `src/uci_loader.py`, `.github/` |

---

## Step-by-Step Git & GitHub Workflow

### Step A: Get the Latest Repository

Before starting any new task, make sure your local repository is up-to-date with `origin/main`:

```bash
# Clone the repository (first time only)
git clone https://github.com/abrilohd/SmartPump-AI.git
cd SmartPump-AI

# Switch to main and pull the latest approved changes
git checkout main
git pull origin main
```

---

### Step B: Create a Task Branch

Never work directly on the `main` branch. Create a descriptive, short-lived task branch from the updated `main`:

```bash
git checkout -b feat/your-task-name
```

**Branch Naming Conventions:**
- `feat/feature-name` — for new assignment implementations (e.g., `feat/sensor-feature-extraction`)
- `fix/bug-name` — for bug fixes (e.g., `fix/kurtosis-zero-division`)
- `docs/doc-name` — for documentation additions (e.g., `docs/training-report`)
- `test/test-name` — for unit or integration tests (e.g., `test/scaler-leakage-check`)
- `refactor/cleanup` — for code restructuring without behavior changes

---

### Step C: Verify Local Changes

As you work, frequently inspect what files you have modified or created:

```bash
# Check status of changed files
git status

# Review line-by-line code changes
git diff
```

Ensure no large sensor files (`*.txt` > 10 MB), temporary caches (`.pytest_cache/`), or virtual environments (`venv/`) are marked for commit.

---

### Step D: Commit Your Work

Commit cohesive, logical units of work with clear commit messages:

```bash
# Stage the files you changed
git add <file1> <file2>

# Commit with a descriptive message and standard prefix
git commit -m "feat: extract 15 cycle-level features for all 17 sensors"
```

**Standard Commit Message Prefixes:**
- `feat:` — new assignment-related implementation
- `fix:` — bug or numerical error fix
- `docs:` — documentation or markdown report updates
- `test:` — adding or updating automated tests
- `refactor:` — necessary code organization or cleanup
- `data:` — dataset documentation or feature metadata updates

---

### Step E: Push Your Branch to GitHub

Push your local task branch to the remote repository:

```bash
git push -u origin feat/your-task-name
```

---

### Step F: Create and Merge a Pull Request

1. Open the repository on GitHub: [https://github.com/abrilohd/SmartPump-AI](https://github.com/abrilohd/SmartPump-AI).
2. Click **Compare & pull request** next to your pushed branch.
3. Set the base branch to `main` and the compare branch to your task branch.
4. Fill out the **Pull Request Template** (`.github/PULL_REQUEST_TEMPLATE.md`):
   - Summarize what was implemented.
   - Specify which Classwork 3.12 requirement is addressed.
   - List the affected files and how they were tested.
5. In the right sidebar, assign at least one fellow team member as a **Reviewer**.
6. Discuss feedback and push additional commits to your branch if changes are requested.
7. Once your team member approves the PR and automated checks pass, click **Merge pull request** (prefer **Squash and merge** or **Rebase and merge**).
8. Delete the merged branch on GitHub.

---

### Step G: Synchronize Local Main After Merging

After your Pull Request is merged into `main`, switch back to your local `main` and pull the latest changes:

```bash
git checkout main
git pull origin main
```

Your local environment is now updated and ready for your next task branch.

---

## Safe Collaboration Guidelines

1. **Do not overwrite teammates' work:** Always coordinate before editing files assigned to another team member.
2. **Jupyter Notebook communication:** When working on notebooks (`notebooks/*.ipynb`), communicate before editing to prevent messy merge conflicts in the notebook JSON structure.
3. **Verify tests before submitting PRs:** Run `python -m pytest tests/ -v` to ensure your changes didn't break existing modules.
4. **No large files:** Never commit raw dataset files (`data/raw/*.txt`). They are excluded via `.gitignore`.
5. **No force-pushing:** Never use `git push --force` on `main` or shared branches.

