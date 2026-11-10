# SmartPump AI — Team Collaboration & GitHub Workflow Guide

## 1. Purpose & Scope

This document defines the team collaboration strategy, branch standards, review protocols, and grading transparency mechanisms for **SmartPump AI** (Classwork 3.12). 

The project is developed collaboratively by a team of **three students** using a single shared GitHub repository:
`https://github.com/abrilohd/SmartPump-AI.git`

The objective is to ensure that work is organized logically, code quality is maintained through peer review, and the instructor can easily inspect individual contributions through Git history, Pull Requests, and repository insights.

---

## 2. Team Structure & Work Distribution

To cover all requirements of Classwork 3.12 without duplicating effort or causing merge conflicts, responsibilities are partitioned into three core areas:

| Student & Role | Primary Responsibilities | Key Deliverables & Files |
|:---|:---|:---|
| **Biruk Ayitenew** (WOUR/0483/16)<br>[@Biruk1997](https://github.com/Biruk1997)<br>*(Member 1: Data Engineering)* | • Raw sensor data acquisition and verification (17 sensors, 2,205 cycles)<br>• Exploratory Data Analysis (EDA) & distribution statistics<br>• Multi-rate sensor alignment & stable-cycle filtering<br>• Cycle-level feature extraction (15 features/sensor = 255 features)<br>• Data quality, missingness, and zero-variance verification | • `src/preprocessing.py`<br>• `src/feature_engineering.py`<br>• `notebooks/01_data_exploration.ipynb`<br>• `notebooks/02_preprocessing.ipynb`<br>• `docs/data_exploration_report.md`<br>• `docs/preprocessing_report.md`<br>• `data/README.md`<br>• `results/eda/` |
| **Abrham Gebremedhin** (WOUR/0082/16)<br>[@abrilohd](https://github.com/abrilohd)<br>*(Member 2: Machine Learning)* | • Implementation of 5 classification algorithms (Decision Tree, Random Forest, SVM, Naïve Bayes, AdaBoost) plus Dummy baseline<br>• Leakage-free cross-validation (5-fold stratified & grouped)<br>• Hyperparameter grid search & model selection<br>• Comprehensive evaluation (Fault precision, Fault recall, Macro F1)<br>• Operational sensitivity evaluation on 756 unstable cycles<br>• Feature importance analysis & visualization generation | • `src/train.py`<br>• `notebooks/03_model_training.ipynb`<br>• `docs/model_training_report.md`<br>• `results/evaluation/*.csv`<br>• `results/visualizations/*.png`<br>• `models/smartpump_best_model.joblib`<br>• `models/model_metadata.json` |
| **Israel Abera** (WOUR/4167/16)<br>[@Isru96](https://github.com/Isru96)<br>*(Member 3: Integration & QA)* | • Project structure setup and architectural integrity<br>• Automated test suite design and maintenance (`tests/`)<br>• UCI programmatic loader implementation & verification (`src/uci_loader.py`)<br>• Project proposal, technical risk register, and conclusions<br>• GitHub collaboration workflows, PR templates, and CONTRIBUTING guide<br>• Root README documentation and submission packaging<br>• Peer review coordination and Git health monitoring | • `README.md`<br>• `CONTRIBUTING.md`<br>• `.github/PULL_REQUEST_TEMPLATE.md`<br>• `docs/team_workflow.md`<br>• `docs/project_proposal.md`<br>• `docs/technical_risks.md`<br>• `docs/dataset_reference.md`<br>• `src/uci_loader.py`<br>• `tests/test_*.py`<br>• `requirements.txt`, `.gitignore` |

---

## 3. Branching Strategy

The repository follows a clean, trunk-based feature branching workflow:

```text
main (production-ready, reviewed code only)
 │
 ├── feat/data-preprocessing     (Member 1 task branch)
 ├── feat/model-training         (Member 2 task branch)
 └── docs/project-integration    (Member 3 task branch)
```

### Branching Rules

1. **`main` is Protected:** No team member commits directly to `main`. All additions and modifications must arrive via a reviewed Pull Request.
2. **Branch from Latest `main`:** Before creating a new branch, always pull the latest commits:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/your-descriptive-branch-name
   ```
3. **Naming Convention:**
   - `feat/<feature-name>`: Implementation of new assignment tasks (e.g., `feat/feature-extraction`, `feat/svm-classifier`).
   - `fix/<bug-name>`: Bug fixes or corrections to existing logic (e.g., `fix/skew-zero-division`).
   - `docs/<doc-name>`: Reports, documentation, or markdown additions (e.g., `docs/model-evaluation-report`).
   - `test/<test-name>`: Unit tests or verification scripts (e.g., `test/variance-threshold-assertions`).
   - `refactor/<cleanup-name>`: Code cleanup and reorganization without changing functional output.
4. **Short-Lived Branches:** Branches should focus on a specific deliverable and be merged promptly once reviewed, rather than accumulating weeks of unmerged changes.

---

## 4. Commit Standards

Team members should make frequent, self-contained commits with clear, descriptive messages using standard prefixes:

| Prefix | Usage Example |
|:---|:---|
| `feat:` | `feat: implement 15 cycle-level statistical features in feature_engineering.py` |
| `fix:` | `fix: prevent precision loss warning in kurtosis calculation for constant signals` |
| `docs:` | `docs: complete model training report with 5-algorithm comparison table` |
| `test:` | `test: add unit test for fit-on-train scaling disjointness` |
| `refactor:` | `refactor: extract baseline model training into reusable helper in train.py` |
| `data:` | `data: update feature metadata catalog with 255 feature descriptions` |

---

## 5. Pull Request & Peer Review Workflow

Every functional addition to the repository must follow this step-by-step review process:

1. **Self-Review & Testing:** Before opening a PR, the author runs the automated test suite locally:
   ```bash
   python -m pytest tests/ -v
   ```
2. **Open Pull Request:** Push the task branch and create a Pull Request on GitHub targeting `main`.
3. **Complete PR Template:** Fill out all sections of `.github/PULL_REQUEST_TEMPLATE.md`, clearly stating which Classwork 3.12 requirement is addressed and how the work was tested.
4. **Request Peer Review:** Assign at least one other team member as a reviewer.
5. **Interactive Review:** The reviewer inspects the diff, checks code quality and tests, and leaves constructive comments if adjustments are needed.
6. **Merge & Cleanup:** Once approved, the PR is merged into `main` (using Squash and Merge or Rebase and Merge to maintain clean history). The author deletes the remote feature branch.
7. **Local Synchronization:** All team members pull the updated `main` into their local environments:
   ```bash
   git checkout main
   git pull origin main
   ```

---

## 6. Teacher Audit & Grading Transparency

The GitHub repository is configured so that an instructor or reviewer can easily verify individual and collaborative contributions:

1. **GitHub Insights / Contributors:**
   Navigate to **Insights → Contributors** on GitHub to view commit frequency, lines of code added/deleted, and timeline distributions for all three accounts.
2. **Commit History (`git log`):**
   Every commit carries the author's individual GitHub identity, verifying who developed each module.
   ```bash
   git log --pretty=format:"%h - %an, %ar : %s"
   ```
3. **Pull Request History:**
   The **Pull Requests (Closed)** tab records the full narrative of collaboration: author proposals, reviewer feedback, iterative improvements, and final approvals.
4. **Separation of Concerns:**
   The clear division between data engineering (`src/preprocessing.py`, `src/feature_engineering.py`), machine learning algorithms (`src/train.py`, `results/`), and system integration (`docs/`, `tests/`, `README.md`) provides immediate visibility into each member's core domain.

---

## 7. Conflict Prevention & Safe Collaboration Guidelines

- **Jupyter Notebook Conflicts:** Simultaneous editing of `.ipynb` files causes difficult JSON merge conflicts. Team members must communicate in advance when working on notebooks. Always execute cells cleanly and ensure outputs are deterministic (`random_state=42`).
- **Never Force-Push:** Do not run `git push --force` or `git push -f` on shared or main branches.
- **Never Commit Raw Datasets:** Raw sensor matrices (`data/raw/*.txt`) total over 500 MB and must never be committed to GitHub. They are safely excluded in `.gitignore`.
- **Preserve Verified Code:** Do not modify working modules authored by another team member without prior coordination.

