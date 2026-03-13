import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & EDA
# ============================================================================
print("=" * 70)
print("SECTION 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv('loan_applications.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\n--- Outcome distribution ---")
print(df['actual_outcome'].value_counts())
print(f"\n--- Rule-based decision distribution ---")
print(df['rule_based_decision'].value_counts())

# Missing values
print(f"\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nNull documented_monthly_income: {df['documented_monthly_income'].isnull().sum()} "
      f"({df['documented_monthly_income'].isnull().mean()*100:.1f}%)")

# Income misrepresentation detection
has_docs = df['documented_monthly_income'].notna()
income_ratio = df.loc[has_docs, 'documented_monthly_income'] / df.loc[has_docs, 'stated_monthly_income']
misrep_mask = has_docs & (df['documented_monthly_income'] < df['stated_monthly_income'] * 0.5)
print(f"\nSuspected misrepresentation (doc < 50% stated): {misrep_mask.sum()} rows "
      f"({misrep_mask.mean()*100:.1f}%)")

# Default rates by key features
print(f"\n--- Default rates by employment status ---")
resolved = df[df['actual_outcome'] != 'ongoing'].copy()
print(resolved.groupby('employment_status')['actual_outcome'].apply(
    lambda x: (x == 'defaulted').mean()
).round(3))

# ---------- EDA Plots ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Exploratory Data Analysis — Loan Applications', fontsize=16, fontweight='bold')

# 1. Outcome distribution
outcome_counts = df['actual_outcome'].value_counts()
colors_outcome = {'repaid': '#2ecc71', 'defaulted': '#e74c3c', 'ongoing': '#95a5a6'}
axes[0, 0].bar(outcome_counts.index, outcome_counts.values, 
               color=[colors_outcome[x] for x in outcome_counts.index], edgecolor='white')
axes[0, 0].set_title('Outcome Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(outcome_counts.values):
    axes[0, 0].text(i, v + 15, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontsize=10)

# 2. Rule-based score distribution by outcome
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df[df['actual_outcome'] == outcome]
    axes[0, 1].hist(subset['rule_based_score'], bins=30, alpha=0.6, label=outcome, color=color)
axes[0, 1].set_title('Rule-Based Score by Outcome', fontweight='bold')
axes[0, 1].set_xlabel('Rule-Based Score')
axes[0, 1].legend()

# 3. Default rate by employment status
emp_default = resolved.groupby('employment_status')['actual_outcome'].apply(
    lambda x: (x == 'defaulted').mean()
)
colors_emp = ['#3498db', '#e67e22', '#e74c3c']
bars = axes[0, 2].bar(emp_default.index, emp_default.values, color=colors_emp, edgecolor='white')
axes[0, 2].set_title('Actual Default Rate by Employment', fontweight='bold')
axes[0, 2].set_ylabel('Default Rate')
for bar, val in zip(bars, emp_default.values):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.1%}', ha='center', fontsize=11)

# 4. Income vs Loan Amount
scatter_df = resolved.sample(min(500, len(resolved)), random_state=42)
for outcome, color, marker in [('repaid', '#2ecc71', 'o'), ('defaulted', '#e74c3c', 'x')]:
    sub = scatter_df[scatter_df['actual_outcome'] == outcome]
    axes[1, 0].scatter(sub['stated_monthly_income'], sub['loan_amount'], 
                       c=color, marker=marker, alpha=0.5, label=outcome, s=20)
axes[1, 0].set_title('Income vs Loan Amount', fontweight='bold')
axes[1, 0].set_xlabel('Stated Monthly Income')
axes[1, 0].set_ylabel('Loan Amount')
axes[1, 0].legend()

# 5. Bank balance distribution
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = resolved[resolved['actual_outcome'] == outcome]
    axes[1, 1].hist(subset['bank_ending_balance'], bins=30, alpha=0.6, label=outcome, color=color)
axes[1, 1].set_title('Bank Balance by Outcome', fontweight='bold')
axes[1, 1].set_xlabel('Bank Ending Balance')
axes[1, 1].legend()

# 6. Rule-based decision vs actual outcome
ct = pd.crosstab(df['rule_based_decision'], df['actual_outcome'], normalize='index')
ct[['repaid', 'defaulted', 'ongoing']].plot(kind='bar', stacked=True, ax=axes[1, 2],
    color=['#2ecc71', '#e74c3c', '#95a5a6'], edgecolor='white')
axes[1, 2].set_title('Rule Decision vs Actual Outcome', fontweight='bold')
axes[1, 2].set_ylabel('Proportion')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)
axes[1, 2].legend(title='Outcome')

plt.tight_layout()
plt.savefig('01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: 01_eda_overview.png")

# ============================================================================
# 2. FEATURE ENGINEERING & DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: FEATURE ENGINEERING & DATA PREPARATION")
print("=" * 70)

# Decision: Exclude 'ongoing' applications
# Rationale: They have no ground truth. Including them would mean either:
#   a) Treating them as 'repaid' (biases toward false negatives)
#   b) Treating them as a separate class (changes the problem)
#   c) Imputing outcomes (introduces noise)
# Excluding is the safest choice, but we note the survivorship bias:
# ongoing apps are likely newer and may differ in risk profile.
print(f"\nExcluding {(df['actual_outcome'] == 'ongoing').sum()} ongoing applications (no ground truth)")
model_df = df[df['actual_outcome'] != 'ongoing'].copy()
print(f"Modeling dataset: {len(model_df)} rows")

# Binary target
model_df['target'] = (model_df['actual_outcome'] == 'defaulted').astype(int)
print(f"Default rate in modeling set: {model_df['target'].mean():.3f}")

# --- Feature Engineering ---
# 1. Income verification features
model_df['has_documentation'] = model_df['documented_monthly_income'].notna().astype(int)
model_df['income_ratio'] = np.where(
    model_df['documented_monthly_income'].notna(),
    model_df['documented_monthly_income'] / model_df['stated_monthly_income'].clip(lower=1),
    np.nan
)
model_df['income_discrepancy'] = np.where(
    model_df['documented_monthly_income'].notna(),
    (model_df['stated_monthly_income'] - model_df['documented_monthly_income']).abs() / model_df['stated_monthly_income'].clip(lower=1),
    -1  # Flag for missing docs
)
model_df['suspected_misrepresentation'] = (
    model_df['documented_monthly_income'].notna() & 
    (model_df['documented_monthly_income'] < model_df['stated_monthly_income'] * 0.5)
).astype(int)

# 2. Affordability features
model_df['loan_to_income_ratio'] = model_df['loan_amount'] / model_df['stated_monthly_income'].clip(lower=1)
model_df['income_covers_loan_3x'] = (model_df['stated_monthly_income'] >= 3 * model_df['loan_amount']).astype(int)

# 3. Account health features
model_df['withdrawal_to_deposit_ratio'] = (
    model_df['monthly_withdrawals'] / model_df['monthly_deposits'].clip(lower=1)
)
model_df['net_monthly_flow'] = model_df['monthly_deposits'] - model_df['monthly_withdrawals']
model_df['balance_to_loan_ratio'] = model_df['bank_ending_balance'] / model_df['loan_amount'].clip(lower=1)
model_df['low_balance'] = (model_df['bank_ending_balance'] < 500).astype(int)

# 4. Employment encoding (ordinal — but we'll discuss this in fairness)
emp_map = {'employed': 2, 'self_employed': 1, 'unemployed': 0}
model_df['employment_encoded'] = model_df['employment_status'].map(emp_map)

# Fill NaN in documented_monthly_income for feature computation
model_df['documented_monthly_income_filled'] = model_df['documented_monthly_income'].fillna(0)

# Feature list
feature_cols = [
    'stated_monthly_income', 'documented_monthly_income_filled', 'loan_amount',
    'bank_ending_balance', 'bank_has_overdrafts', 'bank_has_consistent_deposits',
    'monthly_withdrawals', 'monthly_deposits', 'num_documents_submitted',
    # Engineered features
    'has_documentation', 'income_ratio', 'income_discrepancy',
    'suspected_misrepresentation', 'loan_to_income_ratio', 'income_covers_loan_3x',
    'withdrawal_to_deposit_ratio', 'net_monthly_flow', 'balance_to_loan_ratio',
    'low_balance', 'employment_encoded'
]

# Handle remaining NaN in income_ratio (where docs are missing)
model_df['income_ratio'] = model_df['income_ratio'].fillna(-1)

X = model_df[feature_cols].copy()
y = model_df['target'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {feature_cols}")
print(f"\nClass distribution:\n{y.value_counts()}")

# ============================================================================
# 3. SAVE PROCESSED DATA FOR MODELING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: SAVING PROCESSED DATA")
print("=" * 70)

# Save the full processed dataframe (includes features + metadata for fairness analysis)
model_df.to_csv('processed_data.csv', index=False)
print(f"Saved: processed_data.csv ({len(model_df)} rows)")

# Save feature matrix and target separately
X.to_csv('features.csv', index=False)
y.to_csv('target.csv', index=False)
print(f"Saved: features.csv ({X.shape[0]} rows, {X.shape[1]} features)")
print(f"Saved: target.csv ({len(y)} rows)")
print(f"\nFeature columns saved: {feature_cols}")

print("\n" + "=" * 70)
print("EDA & FEATURE ENGINEERING COMPLETE")
print("Run the modeling script next to train and evaluate models.")
print("=" * 70)