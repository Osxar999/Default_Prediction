## Default Prediction Pipeline

This project simulates a loan underwriting workflow and builds machine‑learning models to predict default. It contains **two parallel pipelines**:

- **Full‑dataset deployment pipeline** (simple, fast, good for prototyping).
- **60/20/20 train–validation–test pipeline** (methodologically correct for reporting and fairness analysis).

Both pipelines share the same synthetic data generator and feature engineering logic.

---

## 0. Setup

- **Python**: `>=3.8`
- Install dependencies (inside a fresh virtualenv is recommended):

```bash
pip install -r requirements.txt
```

All commands below assume the working directory is this folder.

---

## 1. Shared components

### 1.1 Data generation

```bash
python generate_data.py
```

This creates:

- `loan_applications.csv` – synthetic applications with:
  - applicant income (stated / documented, with some misrepresentation),
  - bank account behavior,
  - a **rule‑based score and decision** (existing policy),
  - a simulated **ground‑truth outcome** (`repaid`, `defaulted`, `ongoing`).

### 1.2 EDA & feature engineering

```bash
python eda_feature_engineering.py
```

This script:

- Performs EDA and saves plots to `pictures/01_eda_overview.png`.
- Drops `ongoing` outcomes and creates engineered features such as:
  - `has_documentation`, `income_ratio`, `income_covers_loan_3x`,
  - `suspected_misrepresentation`, `withdrawal_to_deposit_ratio`,
  - `low_balance`, `employment_encoded`.
- Writes core modeling datasets:
  - `data/processed_data.csv` – cleaned labeled rows.
  - `data/features.csv` – feature matrix \(X\).
  - `data/target.csv` – binary target \(y\).

These files are reused by both training pipelines.

---

## 2. Full‑dataset deployment pipeline

**Goal:** Build and evaluate a model using **all available data**, mainly for:

- Fast iteration on model ideas.
- Understanding feature importance / explainability.
- Comparing against the existing rule‑based system.

This pipeline does **not** reserve a strict hold‑out test set, so its metrics are optimistic and mainly for internal understanding, not final reporting.

### 2.1 Train models on all data

```bash
python training.py
```

What it does:

1. Loads `data/processed_data.csv`, `data/features.csv`, `data/target.csv`.
2. Defines several candidate models:
   - Logistic Regression (with scaling),
   - Random Forest,
   - Gradient Boosting,
   - Neural Network (MLP, with scaling).
3. Uses **stratified 5‑fold cross‑validation on the full dataset** to compare AUC and F1.
4. Picks the best model by AUC and **refits it on all rows**.
5. Searches over thresholds on the full dataset to pick the one that **maximizes F1 for the default class**.
6. Compares the ML model against:
   - the rule‑based system (strict: only `denied` = default),
   - a more conservative baseline (denied + flagged).
7. Saves:
   - `model/trained_model.joblib` – final model.
   - `model/scaler.joblib` – scaler (if used).
   - `model/model_outputs.csv` – per‑applicant predictions and probabilities.
   - `model/model_config.json` – config including selected model, threshold, AUC, and `feature_cols`.

**Why this pipeline?**

- Simple to reason about: cross‑validation plus full‑data training.
- Good for prototyping, debugging, and understanding behavior.
- Not ideal for “final numbers” because the threshold and evaluation both use all data.

### 2.2 Evaluation and explainability (full deployment)

```bash
python evaluation.py
```

This script:

- Loads `model/model_outputs.csv`, `data/features.csv`, and the saved model/config.
- Builds side‑by‑side plots comparing:
  - ML vs rule‑based confusion matrices.
  - ROC and precision–recall curves.
  - F1 vs threshold.
  - Score distributions by outcome.
- Computes permutation importance and model‑based feature importance.
- Produces explainability plots:
  - `pictures/02_evaluation.png` – evaluation curves.
  - `pictures/03_explainability.png` – feature importance and permutation importance.
- Prints **sample per‑applicant explanations** that translate features into reasons (e.g. income misrepresentation, low balance, high withdrawal ratio).

### 2.3 Fairness analysis (full deployment)

```bash
python fairness.py
```

This uses the full‑dataset deployment outputs to:

- Compare **approval / denial / default rates** across `employment_status` groups for:
  - the rule‑based policy, and
  - the ML model (using the chosen threshold).
- Retrain a model **without `employment_encoded`** on all data and compare:
  - AUC with vs without the employment feature,
  - group‑level approval rates with vs without.
- Produce a fairness plot:
  - `pictures/04_fairness_full.png`.

**Why this pipeline?**

- Mirrors how the full‑deployment model would behave if trained on all available data.
- Lets you reason about whether including `employment_status` is justified, given differences in observed default rates.
- Still uses full data for both selection and fairness; correct for exploration, but not as strong evidence as a held‑out test analysis.

---

## 3. 60/20/20 train–validation–test split pipeline

**Goal:** Provide a **methodologically sound evaluation** for reporting and fairness:

- Prevents information leakage from test set into model selection or threshold tuning.
- Cleanly separates:
  - training (60%),
  - validation / threshold selection (20%),
  - final evaluation (20% held‑out test).

### 3.1 Split‑based training

```bash
python training_split.py
```

Steps:

1. **Load data**
   - Reads `data/processed_data.csv`, `data/features.csv`, `data/target.csv`.
2. **Create a 60/20/20 stratified split**
   - First: 80/20 split into train+val vs test.
   - Then: split that 80% into 60% train and 20% val (so overall 60/20/20).
   - Keeps consistent default rates across splits.
3. **Model selection (train only)**
   - Same candidate models as in `training.py`, but wrapped in **Pipelines** so scaling happens inside each CV fold.
   - Runs stratified 5‑fold CV on **train only**.
   - Selects the best model by **train‑CV AUC**.
4. **Threshold selection (validation only)**
   - Fits the selected model on **train**.
   - Computes predicted probabilities on **val**.
   - Scans thresholds from 0.10 to 0.89, picks the one that maximizes **validation F1** for defaults.
5. **Honest test evaluation**
   - Evaluates the tuned model and threshold on the **held‑out 20% test set only**.
   - Compares ML vs rule‑based baselines (strict and conservative).
   - Saves:
     - `model_split/test_outputs.csv` – test rows with `target`, `y_prob`, `y_pred`, rule decisions/scores.
     - `model_split/test_indices.npy`, `model_split/trainval_indices.npy` – indices to reproduce the split.
     - `model_split/test_summary_table.csv` – numeric comparison table.
6. **Retrain for deployment (train+val)**
   - Refit the winning model on **train+val** using the chosen threshold.
   - Apply it to **all data** (train+val+test) to get full‑deployment outputs.
   - Save:
     - `model_split/trained_model.joblib`, `model_split/scaler.joblib`.
     - `model_split/model_outputs.csv` – full‑data predictions.
     - `model_split/model_config.json` – including:
       - `feature_cols`, `optimal_threshold`, `auc_roc_test`, `auc_roc_full`,
       - `bl_auc_test`, `bl_auc_full`,
       - `split_sizes` and a summary of test metrics.

**Why this pipeline?**

- Gives **honest, out‑of‑sample test performance** for both metrics and fairness claims.
- Separates concerns:
  - Train+CV on train → pick model.
  - Tune threshold on val only.
  - Report performance and fairness on untouched test only.
- The final deployment model is trained on **train+val**, but decisions about model quality and fairness are based on the held‑out test set.

### 3.2 Split‑based evaluation and explainability

```bash
python evaluation_split.py
```

This script uses the artifacts from `training_split.py` to perform **held‑out test evaluation only**:

- Loads `model_split/test_outputs.csv`.
- Builds **test‑only** plots:
  - Confusion matrices (ML vs rule‑based).
  - ROC and PR curves on the test set.
  - F1 vs threshold on test (diagnostic only; selection was done on val).
  - Score distributions on test.
- Saves plots to:
  - `pictures/02_evaluation_split.png`.
- Writes a numeric comparison table to:
  - `model_split/test_summary_table.csv`.

### 3.3 Split‑based fairness analysis

```bash
python fairness_split.py
```

This script is the split‑aware counterpart of `fairness.py` and focuses on **held‑out test fairness**:

1. **Held‑out test fairness**
   - Loads `model_split/test_outputs.csv`.
   - Computes, per `employment_status` group:
     - approval / denial rates for rule and ML,
     - default rate among approved applicants (rule vs ML).
   - Prints approval gaps (employed vs self‑employed vs unemployed) based **only on the test set**.

2. **Effect of removing `employment_encoded` (train+val only, evaluated on test)**
   - Loads `model_split/trainval_indices.npy`, `model_split/test_indices.npy`,
     `data/target.csv`, and `processed_data.csv`.
   - Rebuilds a model of the same type **without `employment_encoded`**, training on **train+val only**.
   - Evaluates both:
     - AUC with employment vs without on the **test set**.
     - Group‑level approval rates with/without employment on the **same held‑out test rows**.

3. **Plots**
   - Saves fairness plots summarizing test‑based fairness comparisons:
     - `pictures/04_fairness_test.png`.

**Why this pipeline?**

- Aligns fairness conclusions with the same **held‑out test data** used for performance.
- Avoids using full‑dataset metrics (which can be optimistic) as the primary evidence.
- Provides a clean experimental story:
  - “We selected the model and threshold using train+val,
     then evaluated both performance and fairness on a strictly held‑out test set.”

---

## 4. Recommended usage

- For **quick iteration and intuition**:
  - Run: `generate_data.py` → `eda_feature_engineering.py` → `training.py` → `evaluation.py` → `fairness.py`.
- For **final reporting and fairness claims**:
  - Run: `generate_data.py` → `eda_feature_engineering.py` → `training_split.py`
    → `evaluation_split.py` → `fairness_split.py`.

The split pipeline should be your primary source of truth for performance and fairness, while the full‑dataset pipeline is best used for exploration, debugging, and richer explainability views. 

