"""
Student Performance Tracker — Modeling & Evaluation
=====================================================
Baseline : Logistic Regression
Primary  : Random Forest Classifier
Target   : At_Risk (1 = Exam Score < 65, 0 = otherwise)
"""

import matplotlib

# Added matplotlib.use('Agg') so the script runs without needing a display
# non-interactive backend — saves figures and generates the 6 PNGs without opening windows
# I use it when running it on my end, you can uncomment it and use it if you want to run the script without opening figure windows. If you want to see the plots pop up, just leave it commented out.
# matplotlib.use('Agg')  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#F9FAFB", "axes.facecolor": "#F9FAFB",
                     "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN, RED = "#2563EB", "#F97316", "#16A34A", "#DC2626"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv("Data/cleaned_dataset.csv")
print(f"Shape: {df.shape}")
print(df.head())



# 2. CLASS DISTRIBUTION
print("\n" + "=" * 60)
print("2. CLASS DISTRIBUTION")
print("=" * 60)

counts = df['At_Risk'].value_counts().sort_index()
print(counts.to_string())
print(f"\nClass imbalance ratio: {counts[0]/counts[1]:.1f}:1  (Not At-Risk : At-Risk)")

labels = ["Not At-Risk (0)", "At-Risk (1)"]
colors = [GREEN, RED]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(labels, counts.values, color=colors, alpha=.85, width=.4)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')
axes[0].set_title("Class Distribution (Full Dataset)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].spines[["top", "right"]].set_visible(False)
axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=.6))
axes[1].set_title("Class Balance", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig5_class_distribution.png", dpi=150)
plt.show()



# 3. FEATURE & TARGET SPLIT
print("\n" + "=" * 60)
print("3. FEATURE & TARGET SPLIT")
print("=" * 60)

X = df.drop(['At_Risk', 'Exam_Score'], axis=1)
y = df['At_Risk']

print(f"Features : {X.shape[1]}")
print(f"Samples  : {X.shape[0]}")
print(f"\nFeature list:\n{list(X.columns)}")



# 4. TRAIN / TEST SPLIT
print("\n" + "=" * 60)
print("4. TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"At-Risk in train : {y_train.mean():.2%}")
print(f"At-Risk in test  : {y_test.mean():.2%}")


# 5. FEATURE SCALING
print("\n" + "=" * 60)
print("5. FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("StandardScaler applied for Logistic Regression inputs.")


# 6. BASELINE — LOGISTIC REGRESSION
print("\n" + "=" * 60)
print("6. BASELINE — LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)

lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, lr_pred, target_names=["Not At-Risk", "At-Risk"]))


# 7. PRIMARY MODEL — RANDOM FOREST
print("\n" + "=" * 60)
print("7. PRIMARY MODEL — RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    random_state=42, class_weight='balanced', n_jobs=-1
)
rf.fit(X_train, y_train)

rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, rf_pred, target_names=["Not At-Risk", "At-Risk"]))


# 8. METRICS SUMMARY TABLE
print("\n" + "=" * 60)
print("8. EVALUATION METRICS SUMMARY")
print("=" * 60)

def get_metrics(name, y_true, y_pred, y_prob):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
        "Precision": round(precision_score(y_true, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred)    * 100, 2),
        "F1-Score":  round(f1_score(y_true, y_pred)        * 100, 2),
        "AUC-ROC":   round(roc_auc_score(y_true, y_prob)   * 100, 2),
    }

mdf = pd.DataFrame([
    get_metrics("Logistic Regression", y_test, lr_pred, lr_proba),
    get_metrics("Random Forest",       y_test, rf_pred, rf_proba),
]).set_index("Model")

print(mdf.to_string())



# 9 METRICS BAR CHART
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
x, w = np.arange(len(metric_cols)), 0.32

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, mdf.loc["Logistic Regression", metric_cols], w,
       label="Logistic Regression", color=BLUE,   alpha=.88)
ax.bar(x + w/2, mdf.loc["Random Forest",        metric_cols], w,
       label="Random Forest",        color=ORANGE, alpha=.88)

for rect in ax.patches:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.4,
            f"{h:.1f}", ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=.4)
plt.tight_layout()
plt.savefig("fig1_metrics_comparison.png", dpi=150)
plt.show()



# 10  CONFUSION MATRICES
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, name, pred in [
        (axes[0], "Logistic Regression", lr_pred),
        (axes[1], "Random Forest",        rf_pred)]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax,
                linewidths=.5, linecolor="white",
                xticklabels=["Not At-Risk", "At-Risk"],
                yticklabels=["Not At-Risk", "At-Risk"])
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()



# 11 ROC CURVES
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, prob, color in [
        ("Logistic Regression", lr_proba, BLUE),
        ("Random Forest",        rf_proba, ORANGE)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name}  (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=.5, label="Random baseline")
ax.fill_between(*roc_curve(y_test, rf_proba)[:2], alpha=.08, color=ORANGE)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=.4)
plt.tight_layout()
plt.savefig("fig3_roc_curves.png", dpi=150)
plt.show()



# 12 FEATURE IMPORTANCES
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top15 = fi.head(15)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=ORANGE, alpha=.85)
for b in bars:
    ax.text(b.get_width() + .001, b.get_y() + b.get_height()/2,
            f"{b.get_width():.3f}", va='center', fontsize=8.5)
ax.set_xlabel("Importance (Gini)")
ax.set_title("Top 15 Feature Importances – Random Forest", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=.4)
plt.tight_layout()
plt.savefig("fig4_feature_importance.png", dpi=150)
plt.show()

print("\nTop 10 Features:")
print(top15.head(10).to_string())



# 13 (5-Fold, Recall)
print("\n" + "=" * 60)
print("13. 5-FOLD CROSS-VALIDATION (RECALL)")
print("=" * 60)

lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring='recall')
rf_cv = cross_val_score(rf, X_train,    y_train, cv=5, scoring='recall')

cv_df = pd.DataFrame({
    "Logistic Regression": lr_cv,
    "Random Forest":       rf_cv,
}, index=[f"Fold {i+1}" for i in range(5)])

print(cv_df.round(3).to_string())
print(f"\nMean CV Recall — Logistic Regression : {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")
print(f"Mean CV Recall — Random Forest        : {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
cv_df.plot(kind='bar', ax=ax, color=[BLUE, ORANGE], alpha=.85, width=.6)
ax.axhline(lr_cv.mean(), color=BLUE,   linestyle='--', lw=1.5, alpha=.7)
ax.axhline(rf_cv.mean(), color=ORANGE, linestyle='--', lw=1.5, alpha=.7)
ax.set_xticklabels(cv_df.index, rotation=0)
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.05)
ax.set_title("5-Fold CV Recall by Fold", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=.4)
plt.tight_layout()
plt.savefig("fig6_cross_validation.png", dpi=150)
plt.show()



# 14 RESULTS AND DISCUSSION
# adapted the discussion section so the values are not hard coded, but reflect the actual resuts from the metrics table (mdf) and feature importance (fi). This way it will always be accurate regardless of changes to the data or models.
print("\n" + "=" * 60)
print("14. RESULTS & DISCUSSION")
print("=" * 60)
top3 = fi.head(3).index.tolist()
better_accuracy = "Random Forest" if mdf.loc["Random Forest", "Accuracy"] > mdf.loc["Logistic Regression", "Accuracy"] else "Logistic Regression"
better_recall   = "Logistic Regression" if mdf.loc["Logistic Regression", "Recall"] > mdf.loc["Random Forest", "Recall"] else "Random Forest"

print(f"""
Key Findings:
- {better_accuracy} achieves higher overall accuracy ({mdf.loc[better_accuracy, 'Accuracy']:.1f}%) and precision ({mdf.loc[better_accuracy, 'Precision']:.1f}%).
- {better_recall} achieves higher Recall ({mdf.loc['Logistic Regression', 'Recall']:.1f}% vs {mdf.loc['Random Forest', 'Recall']:.1f}%), catching more
  at-risk students — critical for educational intervention use cases.
- Both models achieve AUC-ROC > {mdf['AUC-ROC'].min() / 100:.2f}, indicating strong discriminative ability.
- Top predictors: {', '.join(top3)}.

Recommendation:
  For school deployment where missing a struggling student is costly,
  {better_recall} may be preferred (higher Recall).
  Random Forest is better when fewer false alarms (higher Precision) matter.
""")