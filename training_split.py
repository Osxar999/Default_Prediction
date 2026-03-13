import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ============================================================================
# HELPERS
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "Precision (default)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (default)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (default)": f1_score(y_true, y_pred, zero_division=0),
        "FPR (good applicants wrongly denied)": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "FNR (defaults that slipped through)": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn),
    }
    if y_prob is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_cv_results(results_dict):
    print("\n--- Cross-Validation Results (5-fold Stratified on TRAIN only) ---")
    for name, vals in results_dict.items():
        print(f"\n{name}:")
        print(f"  AUC-ROC: {vals['auc_mean']:.4f} (+/- {vals['auc_std']:.4f})")
        print(f"  F1:      {vals['f1_mean']:.4f} (+/- {vals['f1_std']:.4f})")


def choose_threshold(y_true, y_prob, thresholds):
    best_threshold = None
    best_f1 = -1.0
    scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred_t, zero_division=0)
        scores.append(score)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, scores


# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

data_dir = "data"
model_df = pd.read_csv(os.path.join(data_dir, "processed_data.csv"))
X = pd.read_csv(os.path.join(data_dir, "features.csv"))
y = pd.read_csv(os.path.join(data_dir, "target.csv")).squeeze()

feature_cols = list(X.columns)

print(f"Feature matrix: {X.shape}")
print(f"Target: {len(y)} rows, {y.sum()} defaults ({y.mean():.1%})")


# ============================================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 1: DATA SPLIT")
print("=" * 70)

# First split off test (20%)
all_idx = np.arange(len(X))
trainval_idx, test_idx = train_test_split(
    all_idx, test_size=0.20, random_state=42, stratify=y
)

# Then split train/val from the remaining 80%
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.25,   # 0.25 of 0.80 = 0.20 overall
    random_state=42,
    stratify=y.iloc[trainval_idx]
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

X_val = X.iloc[val_idx].copy()
y_val = y.iloc[val_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

train_df = model_df.iloc[train_idx].copy()
val_df = model_df.iloc[val_idx].copy()
test_df = model_df.iloc[test_idx].copy()

print(f"Data split: {len(y_train)} train, {len(y_val)} validation, {len(y_test)} test")
print(f"Default rates: train={y_train.mean():.1%}, val={y_val.mean():.1%}, test={y_test.mean():.1%}")


# ============================================================================
# 3. MODEL SELECTION (TRAIN ONLY)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: MODEL TRAINING & SELECTION")
print("=" * 70)

# Use pipelines so scaling happens INSIDE CV folds for scaled models.
models = {
    "Logistic Regression (scaled)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=1.0
        ))
    ]),
    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]),
    "Neural Network (scaled)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, pipe in models.items():
    auc_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
    f1_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")

    cv_results[name] = {
        "auc_mean": auc_scores.mean(),
        "auc_std": auc_scores.std(),
        "f1_mean": f1_scores.mean(),
        "f1_std": f1_scores.std(),
    }

print_cv_results(cv_results)

best_name = max(cv_results, key=lambda k: cv_results[k]["auc_mean"])
print(f"\n>>> Best model by TRAIN-CV AUC: {best_name}")

best_model = clone(models[best_name])
best_model.fit(X_train, y_train)

val_prob = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_prob)

print(f"Validation AUC with selected model: {val_auc:.4f}")


# ============================================================================
# 4. THRESHOLD SELECTION (VALIDATION ONLY)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: THRESHOLD SELECTION (VALIDATION ONLY)")
print("=" * 70)

thresholds = np.arange(0.10, 0.90, 0.01)
optimal_threshold, f1_scores_by_thresh = choose_threshold(y_val, val_prob, thresholds)

print(f"Optimal threshold (max validation F1 for default class): {optimal_threshold:.2f}")


# ============================================================================
# 5. HONEST TEST SET EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: HONEST TEST SET EVALUATION")
print("=" * 70)

test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= optimal_threshold).astype(int)

# Save held-out test outputs for honest plotting later
test_output_df = test_df.copy()
test_output_df["target"] = y_test.values
test_output_df["y_prob"] = test_prob
test_output_df["y_pred"] = test_pred
test_output_df.to_csv(os.path.join("model_split", "test_outputs.csv"), index=False)
print(f"Saved: model_split/test_outputs.csv ({len(test_output_df)} held-out test applicants)")

print("\n--- ML Model Performance (held-out test set) ---")
print(classification_report(y_test, test_pred, target_names=["repaid", "defaulted"]))

test_metrics = compute_metrics(y_test, test_pred, test_prob)

# Rule-based baselines on the SAME test set
bl_strict_test = (test_df["rule_based_decision"] == "denied").astype(int)
bl_cons_test = (test_df["rule_based_decision"] != "approved").astype(int)

print("\n--- Rule-Based Baseline (same test set) ---")
print("\nStrict (deny only):")
print(classification_report(y_test, bl_strict_test, target_names=["repaid", "defaulted"]))

print("Conservative (deny + flagged):")
print(classification_report(y_test, bl_cons_test, target_names=["repaid", "defaulted"]))

bl_strict_metrics = compute_metrics(y_test, bl_strict_test)
bl_cons_metrics = compute_metrics(y_test, bl_cons_test)

# Rule-score AUC proxy on test only
bl_prob_test = 1 - test_df["rule_based_score"].values / 100
bl_auc_test = roc_auc_score(y_test, bl_prob_test)
bl_strict_metrics["AUC-ROC"] = bl_auc_test

print("\n" + "=" * 50)
print("COMPARISON TABLE (HONEST TEST SET)")
print("=" * 50)

comparison_test = pd.DataFrame({
    "ML Model": test_metrics,
    "Baseline (strict)": bl_strict_metrics,
    "Baseline (conservative)": bl_cons_metrics
})
print(comparison_test.round(4).to_string())

n_defaults_test = int(y_test.sum())
n_repaid_test = int((y_test == 0).sum())

print(f"\n--- TEST-SET DEPLOYMENT IMPACT ---")
print(
    f"ML Model:            catches {test_metrics['True Positives']}/{n_defaults_test} defaults "
    f"({test_metrics['Recall (default)']:.1%}), wrongly denies "
    f"{test_metrics['False Positives']}/{n_repaid_test} good applicants "
    f"({test_metrics['FPR (good applicants wrongly denied)']:.1%})"
)
print(
    f"Rule (strict):       catches {bl_strict_metrics['True Positives']}/{n_defaults_test} defaults "
    f"({bl_strict_metrics['Recall (default)']:.1%}), wrongly denies "
    f"{bl_strict_metrics['False Positives']}/{n_repaid_test} good applicants "
    f"({bl_strict_metrics['FPR (good applicants wrongly denied)']:.1%})"
)
print(
    f"Rule (conservative): catches {bl_cons_metrics['True Positives']}/{n_defaults_test} defaults "
    f"({bl_cons_metrics['Recall (default)']:.1%}), wrongly denies "
    f"{bl_cons_metrics['False Positives']}/{n_repaid_test} good applicants "
    f"({bl_cons_metrics['FPR (good applicants wrongly denied)']:.1%})"
)


# ============================================================================
# 6. OPTIONAL: RETRAIN FOR DEPLOYMENT ON TRAIN+VAL
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: RETRAIN ON TRAIN+VAL FOR DEPLOYMENT")
print("=" * 70)

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

final_model = clone(models[best_name])
final_model.fit(X_trainval, y_trainval)

# Full-data outputs for downstream explainability scripts if you still want them.
full_prob = final_model.predict_proba(X)[:, 1]
full_pred = (full_prob >= optimal_threshold).astype(int)

baseline_strict_full = (model_df["rule_based_decision"] == "denied").astype(int)
baseline_conservative_full = (model_df["rule_based_decision"] != "approved").astype(int)

full_metrics = compute_metrics(y, full_pred, full_prob)
bl_strict_full_metrics = compute_metrics(y, baseline_strict_full)
bl_cons_full_metrics = compute_metrics(y, baseline_conservative_full)

bl_prob_full = 1 - model_df["rule_based_score"].values / 100
bl_auc_full = roc_auc_score(y, bl_prob_full)
bl_strict_full_metrics["AUC-ROC"] = bl_auc_full

print("\n" + "=" * 50)
print("COMPARISON TABLE (FULL DATA FOR DEPLOYMENT ONLY)")
print("=" * 50)

comparison_full = pd.DataFrame({
    "ML Model": full_metrics,
    "Baseline (strict)": bl_strict_full_metrics,
    "Baseline (conservative)": bl_cons_full_metrics
})
print(comparison_full.round(4).to_string())

print("\nNote: the table above is for deployment-style summary only.")
print("Use the TEST-SET table above as your main reported evaluation.")


# ============================================================================
# 7. SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING MODEL OUTPUTS")
print("=" * 70)

model_dir = "model_split"
os.makedirs(model_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Save final deployment model
joblib.dump(final_model, os.path.join(model_dir, "trained_model.joblib"))

# Save scaler only if pipeline has one
scaler_to_save = None
if "scaler" in final_model.named_steps:
    scaler_to_save = final_model.named_steps["scaler"]

joblib.dump(scaler_to_save, os.path.join(model_dir, "scaler.joblib"))

# Save model outputs on full data for evaluation.py
output_df = model_df.copy()
output_df["target"] = y.values
output_df["y_prob"] = full_prob
output_df["y_pred"] = full_pred
output_df.to_csv(os.path.join(model_dir, "model_outputs.csv"), index=False)

# Save original features too, if your eval script expects them in /data
X.to_csv(os.path.join("data", "features.csv"), index=False)

config = {
    "best_model_name": best_name,
    "optimal_threshold": float(optimal_threshold),
    "feature_cols": feature_cols,
    "auc_roc_test": float(test_metrics["AUC-ROC"]),
    "auc_roc_full": float(full_metrics["AUC-ROC"]),
    "bl_auc_test": float(bl_auc_test),
    "bl_auc_full": float(bl_auc_full),
    "test_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_metrics.items()},
    "validation_auc": float(val_auc),
    "split_sizes": {
        "train": int(len(y_train)),
        "val": int(len(y_val)),
        "test": int(len(y_test)),
    },
    "notes": "Model selected with CV on train only; threshold selected on validation only; final evaluation on held-out test set."
}
with open(os.path.join(model_dir, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Saved: {model_dir}/trained_model.joblib ({best_name})")
print("Saved: {}/scaler.joblib".format(model_dir))
print(f"Saved: {model_dir}/model_outputs.csv ({len(output_df)} applicants)")
print(f"Saved: {model_dir}/model_config.json")
print("Saved: data/features.csv")