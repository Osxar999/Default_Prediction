import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    classification_report,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================================
# HELPERS
# ============================================================================

def get_base_estimator(model):
    """Return underlying estimator if model is a Pipeline, else return model."""
    return getattr(model, "named_steps", {}).get("model", model)


def get_feature_importance_series(model, feature_cols, perm_imp_df=None):
    """
    Return:
      feat_imp: pd.Series sorted ascending
      imp_label: string label for plot title
    """
    base_est = get_base_estimator(model)

    if hasattr(base_est, "coef_"):
        feat_imp = pd.Series(
            np.abs(base_est.coef_[0]), index=feature_cols
        ).sort_values(ascending=True)
        imp_label = "|Coefficient|"

    elif hasattr(base_est, "feature_importances_"):
        feat_imp = pd.Series(
            base_est.feature_importances_, index=feature_cols
        ).sort_values(ascending=True)
        imp_label = "Feature Importance"

    else:
        if perm_imp_df is None:
            raise ValueError(
                "Permutation importance dataframe required when model has no native importances."
            )
        feat_imp = perm_imp_df.set_index("feature")["importance_mean"].sort_values(
            ascending=True
        )
        imp_label = "Permutation Importance"

    return feat_imp, imp_label


def build_eval_dataframe(model_outputs_df, X, feature_cols):
    """
    Make sure the dataframe used for explanations contains all feature columns.
    If a feature is missing in model_outputs_df, pull it from X.
    """
    eval_df = model_outputs_df.copy()

    for col in feature_cols:
        if col not in eval_df.columns:
            eval_df[col] = X[col].values

    return eval_df


def compute_confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
    }


def plot_confusion(ax, y_true, y_pred, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        ax=ax,
        xticklabels=["Repaid", "Defaulted"],
        yticklabels=["Repaid", "Defaulted"],
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def explain_prediction(row, model, feature_cols, threshold):
    """Generate a simple human-readable explanation."""
    features = row[feature_cols].astype(float).values.reshape(1, -1)
    prob = model.predict_proba(features)[0, 1]
    decision = "HIGH RISK (deny)" if prob >= threshold else "LOW RISK (approve)"

    lines = []
    applicant_id = row["applicant_id"] if "applicant_id" in row.index else "<unknown>"
    lines.append(f"Applicant: {applicant_id}")
    lines.append(f"Predicted default probability: {prob:.1%}")
    lines.append(f"Recommendation: {decision}")
    lines.append("---")

    if row.get("suspected_misrepresentation", 0) == 1:
        lines.append("  ⚠ INCOME MISREPRESENTATION: Documented income is <50% of stated")
    if row.get("has_documentation", 1) == 0:
        lines.append("  ⚠ NO DOCUMENTATION: No income documents submitted")
    if row.get("bank_has_overdrafts", 0) == 1:
        lines.append("  ⚠ OVERDRAFTS: Account has overdraft history")
    if row.get("low_balance", 0) == 1:
        lines.append("  ⚠ LOW BALANCE: Bank balance under $500")
    if row.get("withdrawal_to_deposit_ratio", 0) > 0.8:
        lines.append(
            f"  ⚠ HIGH SPENDING: Withdrawals are {row['withdrawal_to_deposit_ratio']:.0%} of deposits"
        )
    if row.get("income_covers_loan_3x", 0) == 0:
        lines.append("  ⚠ AFFORDABILITY: Income does not cover loan 3x")

    if row.get("income_covers_loan_3x", 0) == 1:
        lines.append("  ✓ INCOME BUFFER: Income covers loan 3x+")
    if row.get("bank_has_consistent_deposits", 0) == 1:
        lines.append("  ✓ CONSISTENT DEPOSITS: Regular deposit pattern")
    if row.get("bank_ending_balance", 0) > 2000:
        lines.append(f"  ✓ HEALTHY BALANCE: ${row['bank_ending_balance']:,.0f} ending balance")
    if row.get("has_documentation", 0) == 1 and row.get("suspected_misrepresentation", 0) == 0:
        lines.append("  ✓ VERIFIED INCOME: Documents match stated income")

    return "\n".join(lines)


# ============================================================================
# 1. LOAD EVERYTHING
# ============================================================================

print("=" * 70)
print("LOADING SPLIT-BASED MODEL OUTPUTS")
print("=" * 70)

data_dir = "data"
model_dir = "model_split"
pictures_dir = "pictures"
os.makedirs(pictures_dir, exist_ok=True)

# Load model config (for threshold, names, and AUCs) and features
with open(os.path.join(model_dir, "model_config.json"), "r") as f:
    config = json.load(f)

feature_cols = config["feature_cols"]
optimal_threshold = config["optimal_threshold"]
best_model_name = config["best_model_name"]
ml_auc_test = config.get("auc_roc_test")
split_sizes = config.get("split_sizes", {})

print(f"Model: {best_model_name}")
print(f"Threshold: {optimal_threshold:.2f}")
if ml_auc_test is not None:
    print(f"AUC (held-out test): {ml_auc_test:.4f}")
if split_sizes:
    print(
        f"Split sizes - train: {split_sizes.get('train')}, "
        f"val: {split_sizes.get('val')}, test: {split_sizes.get('test')}"
    )

# Optional held-out test outputs
test_outputs_path = os.path.join(model_dir, "test_outputs.csv")
has_test_outputs = os.path.exists(test_outputs_path)

if has_test_outputs:
    test_df = pd.read_csv(test_outputs_path)
    y_test = test_df["target"]
    y_prob_test = test_df["y_prob"].values
    y_pred_test = test_df["y_pred"].values

    baseline_strict_test = (test_df["rule_based_decision"] == "denied").astype(int)
    baseline_conservative_test = (test_df["rule_based_decision"] != "approved").astype(int)
    bl_prob_test = 1 - test_df["rule_based_score"].values / 100

    print(f"Held-out test rows loaded: {len(test_df)}")
else:
    test_df = None
    print("Held-out test rows not found: model_split/test_outputs.csv")
    print("Add that file in training_split.py to enable honest test-set plots.")


# ============================================================================
# 2. HELD-OUT TEST PLOTS
# ============================================================================

if has_test_outputs:
    print("\n" + "=" * 70)
    print("SECTION 1: HELD-OUT TEST EVALUATION PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Held-Out Test Evaluation — ML Model vs Rule-Based Baseline",
        fontsize=16,
        fontweight="bold",
    )

    # 1. ML confusion matrix
    plot_confusion(
        axes[0, 0],
        y_test,
        y_pred_test,
        f"ML Model on Test (threshold={optimal_threshold:.2f})",
        "Blues",
    )

    # 2. Strict baseline confusion matrix
    plot_confusion(
        axes[0, 1],
        y_test,
        baseline_strict_test,
        "Rule-Based Baseline on Test (strict: deny only)",
        "Oranges",
    )

    # 3. ROC
    fpr_ml_test, tpr_ml_test, _ = roc_curve(y_test, y_prob_test)
    fpr_bl_test, tpr_bl_test, _ = roc_curve(y_test, bl_prob_test)
    auc_ml_test = roc_auc_score(y_test, y_prob_test)
    auc_bl_test = roc_auc_score(y_test, bl_prob_test)

    axes[0, 2].plot(
        fpr_ml_test, tpr_ml_test, linewidth=2, label=f"ML Model (AUC={auc_ml_test:.3f})"
    )
    axes[0, 2].plot(
        fpr_bl_test, tpr_bl_test, "r--", linewidth=2, label=f"Rule-Based (AUC={auc_bl_test:.3f})"
    )
    axes[0, 2].plot([0, 1], [0, 1], "k:", alpha=0.3)
    axes[0, 2].set_title("ROC Curves (Test)", fontweight="bold")
    axes[0, 2].set_xlabel("False Positive Rate")
    axes[0, 2].set_ylabel("True Positive Rate")
    axes[0, 2].legend()

    # 4. Precision-recall
    prec_ml_test, rec_ml_test, _ = precision_recall_curve(y_test, y_prob_test)
    prec_bl_test, rec_bl_test, _ = precision_recall_curve(y_test, bl_prob_test)
    ap_ml_test = average_precision_score(y_test, y_prob_test)
    ap_bl_test = average_precision_score(y_test, bl_prob_test)

    axes[1, 0].plot(
        rec_ml_test, prec_ml_test, linewidth=2, label=f"ML (AP={ap_ml_test:.3f})"
    )
    axes[1, 0].plot(
        rec_bl_test, prec_bl_test, "r--", linewidth=2, label=f"Rule (AP={ap_bl_test:.3f})"
    )
    axes[1, 0].set_title("Precision-Recall Curves (Test)", fontweight="bold")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()

    # 5. Threshold vs F1 on test (diagnostic only; not for selection)
    thresholds = np.arange(0.10, 0.90, 0.01)
    test_f1_scores = [f1_score(y_test, (y_prob_test >= t).astype(int)) for t in thresholds]
    axes[1, 1].plot(thresholds, test_f1_scores, linewidth=2)
    axes[1, 1].axvline(
        optimal_threshold,
        color="r",
        linestyle="--",
        label=f"Chosen on val: {optimal_threshold:.2f}",
    )
    axes[1, 1].set_title("F1 vs Threshold (Test Diagnostic)", fontweight="bold")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("F1 Score (default class)")
    axes[1, 1].legend()

    # 6. Score distributions on test
    axes[1, 2].hist(
        y_prob_test[y_test == 0],
        bins=25,
        alpha=0.6,
        color="#2ecc71",
        label="Repaid",
        density=True,
    )
    axes[1, 2].hist(
        y_prob_test[y_test == 1],
        bins=25,
        alpha=0.6,
        color="#e74c3c",
        label="Defaulted",
        density=True,
    )
    axes[1, 2].axvline(
        optimal_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold: {optimal_threshold:.2f}",
    )
    axes[1, 2].set_title("Score Distribution (Test)", fontweight="bold")
    axes[1, 2].set_xlabel("Predicted Default Probability")
    axes[1, 2].legend()

    plt.tight_layout()
    test_eval_png = os.path.join(pictures_dir, "02_evaluation_split.png")
    plt.savefig(test_eval_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {test_eval_png}")

    print("\n--- Held-Out Test Classification Reports ---")
    print("\nML Model:")
    print(classification_report(y_test, y_pred_test, target_names=["repaid", "defaulted"]))

    print("Rule-Based Strict:")
    print(classification_report(y_test, baseline_strict_test, target_names=["repaid", "defaulted"]))

    print("Rule-Based Conservative:")
    print(classification_report(y_test, baseline_conservative_test, target_names=["repaid", "defaulted"]))


if has_test_outputs:
    ml_test_stats = compute_confusion_metrics(y_test, y_pred_test)
    strict_test_stats = compute_confusion_metrics(y_test, baseline_strict_test)
    cons_test_stats = compute_confusion_metrics(y_test, baseline_conservative_test)

    summary_table = pd.DataFrame(
        {
            "ML Model": {
                "AUC-ROC": roc_auc_score(y_test, y_prob_test),
                "Precision (default)": classification_report(
                    y_test, y_pred_test, output_dict=True, zero_division=0
                )["1"]["precision"],
                "Recall (default)": classification_report(
                    y_test, y_pred_test, output_dict=True, zero_division=0
                )["1"]["recall"],
                "F1 (default)": classification_report(
                    y_test, y_pred_test, output_dict=True, zero_division=0
                )["1"]["f1-score"],
                "FPR": ml_test_stats["fpr"],
                "FNR": ml_test_stats["fnr"],
                "TP": ml_test_stats["tp"],
                "FP": ml_test_stats["fp"],
                "TN": ml_test_stats["tn"],
                "FN": ml_test_stats["fn"],
            },
            "Baseline (strict)": {
                "AUC-ROC": roc_auc_score(y_test, bl_prob_test),
                "Precision (default)": classification_report(
                    y_test, baseline_strict_test, output_dict=True, zero_division=0
                )["1"]["precision"],
                "Recall (default)": classification_report(
                    y_test, baseline_strict_test, output_dict=True, zero_division=0
                )["1"]["recall"],
                "F1 (default)": classification_report(
                    y_test, baseline_strict_test, output_dict=True, zero_division=0
                )["1"]["f1-score"],
                "FPR": strict_test_stats["fpr"],
                "FNR": strict_test_stats["fnr"],
                "TP": strict_test_stats["tp"],
                "FP": strict_test_stats["fp"],
                "TN": strict_test_stats["tn"],
                "FN": strict_test_stats["fn"],
            },
            "Baseline (conservative)": {
                "AUC-ROC": np.nan,
                "Precision (default)": classification_report(
                    y_test, baseline_conservative_test, output_dict=True, zero_division=0
                )["1"]["precision"],
                "Recall (default)": classification_report(
                    y_test, baseline_conservative_test, output_dict=True, zero_division=0
                )["1"]["recall"],
                "F1 (default)": classification_report(
                    y_test, baseline_conservative_test, output_dict=True, zero_division=0
                )["1"]["f1-score"],
                "FPR": cons_test_stats["fpr"],
                "FNR": cons_test_stats["fnr"],
                "TP": cons_test_stats["tp"],
                "FP": cons_test_stats["fp"],
                "TN": cons_test_stats["tn"],
                "FN": cons_test_stats["fn"],
            },
        }
    )

    test_summary_path = os.path.join(model_dir, "test_summary_table.csv")
    summary_table.round(4).to_csv(test_summary_path)
    print(f"\nSaved: {test_summary_path}")

    # This script is dedicated to held-out test evaluation only.
    # Skip full-data deployment plots and explainability.
    print("\n" + "=" * 70)
    print("HELD-OUT TEST EVALUATION COMPLETE")
    print("=" * 70)
    sys.exit(0)
