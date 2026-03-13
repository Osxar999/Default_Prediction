import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    average_precision_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

model_df = pd.read_csv('processed_data.csv')
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').squeeze()

feature_cols = list(X.columns)

print(f"Feature matrix: {X.shape}")
print(f"Target: {len(y)} rows, {y.sum()} defaults ({y.mean():.1%})")

# ============================================================================
# 2. MODEL TRAINING & SELECTION
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: MODEL TRAINING & SELECTION")
print("=" * 70)

# Why stratified k-fold: with 30% defaults, regular splits could put
# 40% defaults in one fold and 20% in another. Stratified keeps it ~30% each.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale features for logistic regression
# LR is sensitive to feature scales — income ranges 500-25000 while
# overdrafts is 0 or 1. Without scaling, big numbers dominate.
# Tree models don't need scaling since they split on thresholds.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Three model candidates:
# - Logistic Regression: simple, interpretable, works well on small data
# - Random Forest: handles non-linear patterns, ensemble of decision trees
# - Gradient Boosting: builds trees sequentially, each fixing the previous one's errors
models = {
    'Logistic Regression (scaled)': ('scaled', LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, C=1.0
    )),
    'Random Forest': ('raw', RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=42, n_jobs=-1
    )),
    'Gradient Boosting': ('raw', GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
}

# class_weight='balanced' upweights the minority class (defaults) so the
# model doesn't just learn to predict 'repaid' for everyone.

print("\n--- Cross-Validation Results (5-fold Stratified) ---")
cv_results = {}
for name, (data_type, model) in models.items():
    X_use = X_scaled if data_type == 'scaled' else X
    auc_scores = cross_val_score(model, X_use, y, cv=skf, scoring='roc_auc')
    f1_scores = cross_val_score(model, X_use, y, cv=skf, scoring='f1')
    cv_results[name] = {'auc_mean': auc_scores.mean(), 'auc_std': auc_scores.std(),
                         'f1_mean': f1_scores.mean(), 'f1_std': f1_scores.std()}
    print(f"\n{name}:")
    print(f"  AUC-ROC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    print(f"  F1:      {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

# Pick the model with the highest AUC
best_name = max(cv_results, key=lambda k: cv_results[k]['auc_mean'])
print(f"\n>>> Best model by AUC: {best_name}")

# Train the winning model on all the data
best_data_type, final_model = models[best_name]
X_final = X_scaled if best_data_type == 'scaled' else X
final_model.fit(X_final, y)

# Get predicted probabilities for every applicant
y_prob = final_model.predict_proba(X_final)[:, 1]

# Find the threshold that maximizes F1 for the default class
# Default threshold of 0.5 isn't always best — we search for the optimal one
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores_by_thresh = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores_by_thresh.append(f1_score(y, y_pred_t))

optimal_threshold = thresholds[np.argmax(f1_scores_by_thresh)]
print(f"\nOptimal threshold (max F1 for default class): {optimal_threshold:.2f}")
y_pred = (y_prob >= optimal_threshold).astype(int)

print("\n" + "=" * 70)
print("SECTION 2: EVALUATION AGAINST BASELINE")
print("=" * 70)

# Convert rule-based decisions to binary predictions for comparison:
# Strict: only 'denied' = predicted default
# Conservative: 'denied' + 'flagged_for_review' = predicted default
baseline_strict = (model_df['rule_based_decision'] == 'denied').astype(int)
baseline_conservative = (model_df['rule_based_decision'] != 'approved').astype(int)

print("\n--- ML Model Performance ---")
print(classification_report(y, y_pred, target_names=['repaid', 'defaulted']))

print("--- Baseline (Rule-Based, Strict: denied=default) ---")
print(classification_report(y, baseline_strict, target_names=['repaid', 'defaulted']))

print("--- Baseline (Rule-Based, Conservative: denied+flagged=default) ---")
print(classification_report(y, baseline_conservative, target_names=['repaid', 'defaulted']))

# Compute detailed metrics for comparison
def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'Precision (default)': precision_score(y_true, y_pred),
        'Recall (default)': recall_score(y_true, y_pred),
        'F1 (default)': f1_score(y_true, y_pred),
        'FPR (good applicants wrongly denied)': fp / (fp + tn),
        'FNR (defaults that slipped through)': fn / (fn + tp),
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }
    if y_prob is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    return metrics

ml_metrics = compute_metrics(y, y_pred, y_prob)
bl_strict_metrics = compute_metrics(y, baseline_strict)
bl_cons_metrics = compute_metrics(y, baseline_conservative)

# Convert rule score to a probability proxy for ROC comparison
# High rule score = low risk, so we flip it: P(default) = 1 - score/100
bl_prob = 1 - model_df['rule_based_score'].values / 100
bl_auc = roc_auc_score(y, bl_prob)
bl_strict_metrics['AUC-ROC'] = bl_auc

print("\n" + "=" * 50)
print("COMPARISON TABLE")
print("=" * 50)
comparison = pd.DataFrame({
    'ML Model': ml_metrics,
    'Baseline (strict)': bl_strict_metrics,
    'Baseline (conservative)': bl_cons_metrics
})
print(comparison.round(4).to_string())

# Deployment impact — what changes if we switch from rule-based to ML?
print("\n--- DEPLOYMENT IMPACT ANALYSIS ---")
n_defaults = y.sum()
n_repaid = (y == 0).sum()

ml_caught = ml_metrics['True Positives']
ml_wrongly_denied = ml_metrics['False Positives']
bl_caught = bl_strict_metrics['True Positives']
bl_wrongly_denied = bl_strict_metrics['False Positives']

print(f"\nTotal defaults in data: {n_defaults}")
print(f"Total repaid in data: {n_repaid}")
print(f"\nML Model:      catches {ml_caught}/{n_defaults} defaults ({ml_caught/n_defaults:.1%}), "
      f"wrongly denies {ml_wrongly_denied}/{n_repaid} good applicants ({ml_wrongly_denied/n_repaid:.1%})")
print(f"Rule (strict): catches {bl_caught}/{n_defaults} defaults ({bl_caught/n_defaults:.1%}), "
      f"wrongly denies {bl_wrongly_denied}/{n_repaid} good applicants ({bl_wrongly_denied/n_repaid:.1%})")
print(f"\nNet improvement:")
print(f"  Additional defaults caught: {ml_caught - bl_caught}")
print(f"  Additional good applicants wrongly denied: {ml_wrongly_denied - bl_wrongly_denied}")