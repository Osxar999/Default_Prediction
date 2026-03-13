import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================================
# HELPERS
# ============================================================================

def get_base_estimator(model):
    return getattr(model, "named_steps", {}).get("model", model)


def rebuild_model_without_feature(final_model, use_scaling, random_state=42):
    """
    Rebuild a model of the same estimator class, but to be refit on a reduced feature set.
    Works for pipeline and non-pipeline estimators.
    """
    base_est = get_base_estimator(final_model)
    new_est = clone(base_est)

    if use_scaling:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", new_est)
        ])
    return new_est


def safe_mean_boolean(mask):
    return float(np.mean(mask)) if len(mask) > 0 else np.nan


def group_fairness_table(df, prob_col, threshold):
    """
    Compute fairness stats by employment group.
    Expects df to include:
      - employment_status
      - target
      - rule_based_decision
      - probability column prob_col
    """
    rows = []

    for emp_status in ["employed", "self_employed", "unemployed"]:
        group = df[df["employment_status"] == emp_status].copy()

        if len(group) == 0:
            continue

        actual_default_rate = group["target"].mean()

        rule_approve = (group["rule_based_decision"] == "approved")
        rule_deny = (group["rule_based_decision"] == "denied")

        ml_approve = group[prob_col] < threshold
        ml_deny = group[prob_col] >= threshold

        rule_default_in_approved = (
            group.loc[rule_approve, "target"].mean() if rule_approve.sum() > 0 else np.nan
        )
        ml_default_in_approved = (
            group.loc[ml_approve, "target"].mean() if ml_approve.sum() > 0 else np.nan
        )

        rows.append({
            "Employment Status": emp_status,
            "N": len(group),
            "Actual Default Rate": actual_default_rate,
            "Rule: Approval Rate": rule_approve.mean(),
            "Rule: Denial Rate": rule_deny.mean(),
            "Rule: Default in Approved": rule_default_in_approved,
            "ML: Approval Rate": ml_approve.mean(),
            "ML: Denial Rate": ml_deny.mean(),
            "ML: Default in Approved": ml_default_in_approved,
        })

    return pd.DataFrame(rows)


# ============================================================================
# 1. LOAD SPLIT-BASED ARTIFACTS
# ============================================================================

print("=" * 70)
print("LOADING SPLIT-BASED MODEL OUTPUTS")
print("=" * 70)

data_dir = "data"
model_dir = "model_split"
pictures_dir = "pictures"
os.makedirs(pictures_dir, exist_ok=True)

model_path = os.path.join(model_dir, "trained_model.joblib")
config_path = os.path.join(model_dir, "model_config.json")
test_outputs_path = os.path.join(model_dir, "test_outputs.csv")
features_path = os.path.join(data_dir, "features.csv")

if not os.path.exists(test_outputs_path):
    raise FileNotFoundError(
        "Missing model_split/test_outputs.csv. Save held-out test outputs in training_split.py first."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError("Missing model_split/trained_model.joblib")

if not os.path.exists(config_path):
    raise FileNotFoundError("Missing model_split/model_config.json")

if not os.path.exists(features_path):
    raise FileNotFoundError("Missing data/features.csv")

test_df = pd.read_csv(test_outputs_path)
X_all = pd.read_csv(features_path)

final_model = joblib.load(model_path)

with open(config_path, "r") as f:
    config = json.load(f)

feature_cols = config["feature_cols"]
optimal_threshold = config["optimal_threshold"]
best_model_name = config["best_model_name"]
split_sizes = config.get("split_sizes", {})
test_auc = config.get("auc_roc_test", None)

print(f"Model: {best_model_name}")
print(f"Threshold: {optimal_threshold:.2f}")
if test_auc is not None:
    print(f"Held-out test AUC: {test_auc:.4f}")
print(f"Held-out test applicants: {len(test_df)}")
if split_sizes:
    print(
        f"Split sizes - train: {split_sizes.get('train')}, "
        f"val: {split_sizes.get('val')}, test: {split_sizes.get('test')}"
    )

# Check required fields
required_cols = [
    "employment_status",
    "target",
    "rule_based_decision",
    "y_prob",
]
missing = [c for c in required_cols if c not in test_df.columns]
if missing:
    raise ValueError(f"test_outputs.csv is missing required columns: {missing}")


# ============================================================================
# 2. FAIRNESS ANALYSIS ON HELD-OUT TEST SET
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 1: FAIRNESS ANALYSIS ON HELD-OUT TEST SET")
print("=" * 70)

fairness_df = group_fairness_table(test_df, prob_col="y_prob", threshold=optimal_threshold)

print("\n--- Fairness Comparison: Employment Status Groups (Held-Out Test Set) ---")
print(fairness_df.round(4).to_string(index=False))

emp_approve_ml = fairness_df.set_index("Employment Status")["ML: Approval Rate"]
emp_approve_rule = fairness_df.set_index("Employment Status")["Rule: Approval Rate"]

print(f"\n--- Approval Gap Analysis (Held-Out Test Set) ---")
print(
    f"Rule approval gap (employed vs self_employed): "
    f"{emp_approve_rule.get('employed', 0) - emp_approve_rule.get('self_employed', 0):.3f}"
)
print(
    f"Rule approval gap (employed vs unemployed):    "
    f"{emp_approve_rule.get('employed', 0) - emp_approve_rule.get('unemployed', 0):.3f}"
)
print(
    f"\nML approval gap (employed vs self_employed):   "
    f"{emp_approve_ml.get('employed', 0) - emp_approve_ml.get('self_employed', 0):.3f}"
)
print(
    f"ML approval gap (employed vs unemployed):      "
    f"{emp_approve_ml.get('employed', 0) - emp_approve_ml.get('unemployed', 0):.3f}"
)


# ============================================================================
# 3. RETRAIN WITHOUT EMPLOYMENT FEATURE (TRAIN+VAL ONLY), EVALUATE ON TEST
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: IMPACT OF REMOVING EMPLOYMENT FEATURE")
print("=" * 70)

# Strongly recommended saved artifacts from training_split.py:
trainval_idx_path = os.path.join(model_dir, "trainval_indices.npy")
test_idx_path = os.path.join(model_dir, "test_indices.npy")
target_path = os.path.join(data_dir, "target.csv")
processed_path = os.path.join(data_dir, "processed_data.csv")

if not os.path.exists(trainval_idx_path) or not os.path.exists(test_idx_path):
    raise FileNotFoundError(
        "Missing trainval_indices.npy / test_indices.npy in model_split/. "
        "Save split indices in training_split.py so fairness retraining can follow the honest split."
    )

if not os.path.exists(target_path):
    raise FileNotFoundError("Missing data/target.csv for fairness retraining.")

if not os.path.exists(processed_path):
    raise FileNotFoundError(
        "Missing processed_data.csv. Needed here to rebuild test metadata cleanly."
    )

trainval_idx = np.load(trainval_idx_path)
test_idx = np.load(test_idx_path)

y_all = pd.read_csv(target_path).squeeze()
model_df_all = pd.read_csv(processed_path)

feature_cols_no_emp = [c for c in feature_cols if c != "employment_encoded"]

X_trainval_no_emp = X_all.iloc[trainval_idx][feature_cols_no_emp].copy()
y_trainval = y_all.iloc[trainval_idx].copy()

X_test_no_emp = X_all.iloc[test_idx][feature_cols_no_emp].copy()
y_test = y_all.iloc[test_idx].copy()

# Figure out whether the winning model uses scaling
use_scaling = "scaled" in best_model_name.lower()

model_no_emp = rebuild_model_without_feature(
    final_model=final_model,
    use_scaling=use_scaling
)

model_no_emp.fit(X_trainval_no_emp, y_trainval)
y_prob_no_emp_test = model_no_emp.predict_proba(X_test_no_emp)[:, 1]

# Attach to held-out test dataframe
test_df_no_emp = test_df.copy()
test_df_no_emp["y_prob_no_emp"] = y_prob_no_emp_test

fairness_no_emp_rows = []
for emp_status in ["employed", "self_employed", "unemployed"]:
    group = test_df_no_emp[test_df_no_emp["employment_status"] == emp_status].copy()
    if len(group) == 0:
        continue

    fairness_no_emp_rows.append({
        "Employment Status": emp_status,
        "Actual Default Rate": group["target"].mean(),
        "ML (with emp): Approval Rate": (group["y_prob"] < optimal_threshold).mean(),
        "ML (no emp): Approval Rate": (group["y_prob_no_emp"] < optimal_threshold).mean(),
    })

fairness_no_emp_df = pd.DataFrame(fairness_no_emp_rows)

print("\n--- Fairness WITH vs WITHOUT employment feature (Held-Out Test Set) ---")
print(fairness_no_emp_df.round(4).to_string(index=False))

auc_with = roc_auc_score(y_test, test_df_no_emp["y_prob"])
auc_without = roc_auc_score(y_test, test_df_no_emp["y_prob_no_emp"])

print(f"\nHeld-out test AUC with employment:    {auc_with:.4f}")
print(f"Held-out test AUC without employment: {auc_without:.4f}")
print(f"AUC drop on held-out test:            {auc_with - auc_without:.4f}")

# Safer interpretation text
fd = fairness_df.set_index("Employment Status")
emp_default = fd.loc["employed", "Actual Default Rate"] if "employed" in fd.index else np.nan
self_default = fd.loc["self_employed", "Actual Default Rate"] if "self_employed" in fd.index else np.nan
unemp_default = fd.loc["unemployed", "Actual Default Rate"] if "unemployed" in fd.index else np.nan

print(f"""
INTERPRETATION (based on held-out test set):
- The fairness comparison should be judged on the held-out test set, not on the full-data deployment model.
- Removing employment_encoded changes performance by {auc_with - auc_without:.4f} AUC on the held-out test set.
- Employed vs self-employed groups should be compared to see whether different approval rates are justified by meaningful default-rate differences.
- Observed test default rates:
    employed:      {emp_default:.1%}
    self_employed: {self_default:.1%}
    unemployed:    {unemp_default:.1%}
- If approval gaps are large while default-rate differences are small, that is evidence of unfair treatment.
- Even if unemployed applicants show higher default rates, you may still argue that direct use of employment status is a governance risk if similar information is already captured through income, deposits, and balance signals.

RECOMMENDATION:
Use the held-out test fairness results as the primary evidence.
If removing employment causes only a small test-AUC drop while reducing unjustified approval gaps, removing it is a defensible policy choice.
""")


# ============================================================================
# 4. FAIRNESS PLOTS (HELD-OUT TEST SET)
# ============================================================================

print("=" * 70)
print("SECTION 3: FAIRNESS PLOTS (HELD-OUT TEST SET)")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Fairness Analysis — Employment Status Groups (Held-Out Test Set)",
    fontsize=16,
    fontweight="bold"
)

emp_labels = ["employed", "self_employed", "unemployed"]
x = np.arange(len(emp_labels))
width = 0.30

fd = fairness_df.set_index("Employment Status")
fn = fairness_no_emp_df.set_index("Employment Status")

# 1. Approval rates vs actual repayment
ax = axes[0]
ax.bar(
    x - width,
    [fd.loc[g, "Rule: Approval Rate"] if g in fd.index else np.nan for g in emp_labels],
    width,
    label="Rule-Based",
    alpha=0.85,
)
ax.bar(
    x,
    [fd.loc[g, "ML: Approval Rate"] if g in fd.index else np.nan for g in emp_labels],
    width,
    label="ML Model",
    alpha=0.85,
)
ax.bar(
    x + width,
    [1 - fd.loc[g, "Actual Default Rate"] if g in fd.index else np.nan for g in emp_labels],
    width,
    label="Actual Repayment Rate",
    alpha=0.85,
)
ax.set_title("Approval Rates vs Actual Repayment", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(emp_labels)
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.10)
ax.legend()

# 2. Default rate among approved
ax = axes[1]
ax.bar(
    x - width / 2,
    [fd.loc[g, "Rule: Default in Approved"] if g in fd.index else np.nan for g in emp_labels],
    width,
    label="Rule-Based",
    alpha=0.85,
)
ax.bar(
    x + width / 2,
    [fd.loc[g, "ML: Default in Approved"] if g in fd.index else np.nan for g in emp_labels],
    width,
    label="ML Model",
    alpha=0.85,
)
ax.set_title("Default Rate Among Approved", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(emp_labels)
ax.set_ylabel("Default Rate in Approved")
ax.legend()

# 3. With vs without employment feature
ax = axes[2]
ax.bar(
    x - width / 2,
    [fn.loc[g, "ML (with emp): Approval Rate"] if g in fn.index else np.nan for g in emp_labels],
    width,
    label="With Employment",
    alpha=0.85,
)
ax.bar(
    x + width / 2,
    [fn.loc[g, "ML (no emp): Approval Rate"] if g in fn.index else np.nan for g in emp_labels],
    width,
    label="Without Employment",
    alpha=0.85,
)
ax.set_title("ML Approval Rate: With vs Without Employment", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(emp_labels)
ax.set_ylabel("Approval Rate")
ax.set_ylim(0, 1.10)
ax.legend()

plt.tight_layout()

fairness_png_path = os.path.join(pictures_dir, "04_fairness_split.png")
plt.savefig(fairness_png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fairness_png_path}")


# ============================================================================
# 5. OPTIONAL SECONDARY NOTE ABOUT DEPLOYMENT
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: PRODUCTION CONSIDERATIONS")
print("=" * 70)

print(
    "Primary fairness claims should come from the held-out test set. "
    "The retrained full-data deployment model can be used for implementation, "
    "but not as the main evidence for fairness conclusions."
)

print("\nFAIRNESS ANALYSIS COMPLETE")