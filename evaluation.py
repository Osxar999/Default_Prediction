import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, average_precision_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD EVERYTHING
# ============================================================================
print("=" * 70)
print("LOADING MODEL OUTPUTS")
print("=" * 70)

data_dir = "data"
model_dir = "model"

model_df = pd.read_csv(os.path.join(model_dir, "model_outputs.csv"))
X = pd.read_csv(os.path.join(data_dir, "features.csv"))
y = model_df['target']
y_prob = model_df['y_prob'].values
y_pred = model_df['y_pred'].values

final_model = joblib.load(os.path.join(model_dir, "trained_model.joblib"))
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

with open(os.path.join(model_dir, "model_config.json"), 'r') as f:
    config = json.load(f)

feature_cols = config['feature_cols']
optimal_threshold = config['optimal_threshold']
best_data_type = config['best_data_type']
bl_auc = config['bl_auc']

X_final = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index) if best_data_type == 'scaled' else X

print(f"Model: {config['best_model_name']}")
print(f"Threshold: {optimal_threshold:.2f}")
print(f"AUC: {config['auc_roc']:.4f}")
print(f"Applicants: {len(model_df)}")

# ============================================================================
# 2. EVALUATION PLOTS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: EVALUATION PLOTS")
print("=" * 70)

baseline_strict = (model_df['rule_based_decision'] == 'denied').astype(int)
bl_prob = 1 - model_df['rule_based_score'].values / 100

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Model Evaluation — ML Model vs Rule-Based Baseline', fontsize=16, fontweight='bold')

# 1. Confusion Matrix — ML Model
cm_ml = confusion_matrix(y, y_pred)
sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Repaid', 'Defaulted'], yticklabels=['Repaid', 'Defaulted'])
axes[0, 0].set_title(f'ML Model (threshold={optimal_threshold:.2f})', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. Confusion Matrix — Baseline
cm_bl = confusion_matrix(y, baseline_strict)
sns.heatmap(cm_bl, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
            xticklabels=['Repaid', 'Defaulted'], yticklabels=['Repaid', 'Defaulted'])
axes[0, 1].set_title('Rule-Based Baseline (strict: deny only)', fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. ROC Curves
fpr_ml, tpr_ml, _ = roc_curve(y, y_prob)
fpr_bl, tpr_bl, _ = roc_curve(y, bl_prob)
ml_auc = roc_auc_score(y, y_prob)
axes[0, 2].plot(fpr_ml, tpr_ml, 'b-', linewidth=2,
                label=f'ML Model (AUC={ml_auc:.3f})')
axes[0, 2].plot(fpr_bl, tpr_bl, 'r--', linewidth=2,
                label=f'Rule-Based (AUC={bl_auc:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k:', alpha=0.3)
axes[0, 2].set_title('ROC Curves', fontweight='bold')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].legend()

# 4. Precision-Recall Curve
prec_ml, rec_ml, _ = precision_recall_curve(y, y_prob)
prec_bl, rec_bl, _ = precision_recall_curve(y, bl_prob)
axes[1, 0].plot(rec_ml, prec_ml, 'b-', linewidth=2,
                label=f'ML (AP={average_precision_score(y, y_prob):.3f})')
axes[1, 0].plot(rec_bl, prec_bl, 'r--', linewidth=2,
                label=f'Rule (AP={average_precision_score(y, bl_prob):.3f})')
axes[1, 0].set_title('Precision-Recall Curves', fontweight='bold')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()

# 5. Threshold vs F1
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores_by_thresh = [f1_score(y, (y_prob >= t).astype(int)) for t in thresholds]
axes[1, 1].plot(thresholds, f1_scores_by_thresh, 'b-', linewidth=2)
axes[1, 1].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
axes[1, 1].set_title('F1 Score vs Classification Threshold', fontweight='bold')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('F1 Score (default class)')
axes[1, 1].legend()

# 6. Score distributions
axes[1, 2].hist(y_prob[y == 0], bins=40, alpha=0.6, color='#2ecc71', label='Repaid', density=True)
axes[1, 2].hist(y_prob[y == 1], bins=40, alpha=0.6, color='#e74c3c', label='Defaulted', density=True)
axes[1, 2].axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
axes[1, 2].set_title('ML Model Score Distribution', fontweight='bold')
axes[1, 2].set_xlabel('Predicted Default Probability')
axes[1, 2].legend()

plt.tight_layout()

pictures_dir = "pictures"
os.makedirs(pictures_dir, exist_ok=True)

eval_png_path = os.path.join(pictures_dir, "02_evaluation_test.png")
plt.savefig(eval_png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {eval_png_path}")

# ============================================================================
# 3. EXPLAINABILITY
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MODEL EXPLAINABILITY")
print("=" * 70)

# Permutation importance: shuffle one feature, measure AUC drop
perm_imp = permutation_importance(final_model, X_final, y, n_repeats=10,
                                   random_state=42, scoring='roc_auc')
perm_imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)

print("\n--- Permutation Importance (AUC-ROC) ---")
print(perm_imp_df.to_string(index=False))

# Model coefficients
if hasattr(final_model, 'coef_'):
    feat_imp = pd.Series(np.abs(final_model.coef_[0]), index=feature_cols).sort_values(ascending=True)
    print("\n--- Feature Importances (|coefficient|) ---")
    print(feat_imp.sort_values(ascending=False).round(4).to_string())
elif hasattr(final_model, 'feature_importances_'):
    feat_imp = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    print("\n--- Feature Importances (Gini/gain-based) ---")
    print(feat_imp.sort_values(ascending=False).round(4).to_string())
else:
    feat_imp = perm_imp_df.set_index('feature')['importance_mean'].sort_values(ascending=True)

# ---------- Explainability Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Model Explainability', fontsize=16, fontweight='bold')

feat_imp.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
imp_label = '|Coefficient|' if hasattr(final_model, 'coef_') else 'Gain-based'
axes[0].set_title(f'Feature Importances ({imp_label})', fontweight='bold')
axes[0].set_xlabel('Importance')

perm_sorted = perm_imp_df.sort_values('importance_mean', ascending=True)
axes[1].barh(perm_sorted['feature'], perm_sorted['importance_mean'],
             xerr=perm_sorted['importance_std'], color='coral', edgecolor='white')
axes[1].set_title('Permutation Importance (AUC-ROC)', fontweight='bold')
axes[1].set_xlabel('Mean Decrease in AUC')

plt.tight_layout()

expl_png_path = os.path.join(pictures_dir, "03_explainability.png")
plt.savefig(expl_png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {expl_png_path}")

# --- Individual Prediction Explanations ---
print("\n--- Sample Individual Explanations ---")

def explain_prediction(row, model, feature_cols, threshold, scaler_obj=None):
    """Generate a human-readable explanation for why someone was flagged or approved."""
    features = row[feature_cols].values.reshape(1, -1).astype(float)
    if scaler_obj is not None:
        features = scaler_obj.transform(features)
    prob = model.predict_proba(features)[0, 1]
    decision = "HIGH RISK (deny)" if prob >= threshold else "LOW RISK (approve)"

    lines = []
    lines.append(f"Applicant: {row['applicant_id']}")
    lines.append(f"Predicted default probability: {prob:.1%}")
    lines.append(f"Recommendation: {decision}")
    lines.append(f"---")

    if row.get('suspected_misrepresentation', 0) == 1:
        lines.append(f"  ⚠ INCOME MISREPRESENTATION: Documented income is <50% of stated")
    if row.get('has_documentation', 1) == 0:
        lines.append(f"  ⚠ NO DOCUMENTATION: No income documents submitted")
    if row.get('bank_has_overdrafts', False):
        lines.append(f"  ⚠ OVERDRAFTS: Account has overdraft history")
    if row.get('low_balance', 0) == 1:
        lines.append(f"  ⚠ LOW BALANCE: Bank balance under $500")
    if row.get('withdrawal_to_deposit_ratio', 0) > 0.8:
        lines.append(f"  ⚠ HIGH SPENDING: Withdrawals are {row['withdrawal_to_deposit_ratio']:.0%} of deposits")
    if row.get('income_covers_loan_3x', 0) == 0:
        lines.append(f"  ⚠ AFFORDABILITY: Income does not cover loan 3x")

    if row.get('income_covers_loan_3x', 0) == 1:
        lines.append(f"  ✓ INCOME BUFFER: Income covers loan 3x+")
    if row.get('bank_has_consistent_deposits', False):
        lines.append(f"  ✓ CONSISTENT DEPOSITS: Regular deposit pattern")
    if row.get('bank_ending_balance', 0) > 2000:
        lines.append(f"  ✓ HEALTHY BALANCE: ${row['bank_ending_balance']:,.0f} ending balance")
    if row.get('has_documentation', 0) == 1 and row.get('suspected_misrepresentation', 0) == 0:
        lines.append(f"  ✓ VERIFIED INCOME: Documents match stated income")

    return "\n".join(lines)

scaler_to_pass = scaler if best_data_type == 'scaled' else None
sample_default = model_df[model_df['target'] == 1].iloc[0]
sample_repaid = model_df[model_df['target'] == 0].iloc[0]
print("\n" + explain_prediction(sample_default, final_model, feature_cols, optimal_threshold, scaler_to_pass))
print("\n" + explain_prediction(sample_repaid, final_model, feature_cols, optimal_threshold, scaler_to_pass))

print("\n" + "=" * 70)
print("EVALUATION & EXPLAINABILITY COMPLETE")