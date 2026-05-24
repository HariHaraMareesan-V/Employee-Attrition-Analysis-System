"""
03_visualizations.py
Employee Attrition Dashboard + ML Visualizations
"""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("charts", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════

GREEN  = "#1B5E20"
LGREEN = "#388E3C"
RED    = "#C62828"
BLUE   = "#1F4E79"
LBLUE  = "#2E75B6"
GRAY   = "#757575"
ACCENT = "#F1F8E9"

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

df = pd.read_csv("data/hr_attrition.csv")

fi = pd.read_csv("data/feature_importance.csv")

mc = pd.read_csv("data/model_comparison.csv")

pr = pd.read_csv("data/predictions.csv")

# ADD RISK LEVEL IF MISSING
if "Risk_Level" not in pr.columns:

    pr["Risk_Level"] = pd.cut(

        pr["Attrition_Probability"],

        bins=[0, 0.3, 0.6, 1.0],

        labels=["Low", "Medium", "High"]
    )

# ═══════════════════════════════════════════════════════════════
# DASHBOARD PAGE 1
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 26), facecolor=ACCENT)

fig.suptitle(
    "Employee Attrition Analysis Dashboard",
    fontsize=26,
    fontweight="bold",
    color=GREEN
)

gs = gridspec.GridSpec(
    4,
    2,
    figure=fig,
    hspace=0.45,
    wspace=0.3
)

# ──────────────────────────────────────────────────────────────
# KPI SECTION
# ──────────────────────────────────────────────────────────────

total = len(df)

left = (df["Attrition"] == "Yes").sum()

stay = total - left

rate = (left / total) * 100

# KPI CARD
ax0 = fig.add_subplot(gs[0, :])

ax0.axis("off")

kpi_text = f"""
Total Employees : {total:,}

Employees Left  : {left:,}

Employees Stayed: {stay:,}

Attrition Rate  : {rate:.2f}%
"""

ax0.text(
    0.5,
    0.5,
    kpi_text,
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=GREEN
)

# ──────────────────────────────────────────────────────────────
# ATTRITION PIE
# ──────────────────────────────────────────────────────────────

ax1 = fig.add_subplot(gs[1, 0])

ax1.pie(
    [stay, left],
    labels=["Stayed", "Left"],
    autopct="%1.1f%%",
    colors=[LGREEN, RED]
)

ax1.set_title(
    "Employee Attrition Distribution",
    fontsize=14,
    fontweight="bold"
)

# ──────────────────────────────────────────────────────────────
# DEPARTMENT ATTRITION
# ──────────────────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[1, 1])

dept = df.groupby("Department").apply(

    lambda x:
    (x["Attrition"] == "Yes").mean() * 100

).reset_index(name="Attrition_Rate")

ax2.bar(
    dept["Department"],
    dept["Attrition_Rate"],
    color=[BLUE, GREEN, RED]
)

ax2.set_title(
    "Attrition Rate by Department",
    fontsize=14,
    fontweight="bold"
)

ax2.set_ylabel("Attrition %")

# ──────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────

ax3 = fig.add_subplot(gs[2, :])

top15 = fi.head(15)

ax3.barh(
    top15["Feature"][::-1],
    top15["Importance"][::-1],
    color=LGREEN
)

ax3.set_title(
    "Top Feature Importances",
    fontsize=14,
    fontweight="bold"
)

# ──────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────────────────────

ax4 = fig.add_subplot(gs[3, 0])

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

x = np.arange(len(metrics))

width = 0.25

for i, (_, row) in enumerate(mc.iterrows()):

    vals = [float(row[m]) for m in metrics]

    ax4.bar(
        x + i * width,
        vals,
        width,
        label=row["Model"]
    )

ax4.set_xticks(x + width)

ax4.set_xticklabels(metrics)

ax4.set_ylim(0, 1.1)

ax4.legend()

ax4.set_title(
    "Model Comparison",
    fontsize=14,
    fontweight="bold"
)

# ──────────────────────────────────────────────────────────────
# RISK DISTRIBUTION
# ──────────────────────────────────────────────────────────────

ax5 = fig.add_subplot(gs[3, 1])

risk_counts = pr["Risk_Level"].value_counts()

risk_counts = risk_counts.reindex(
    ["Low", "Medium", "High"]
)

ax5.bar(
    risk_counts.index,
    risk_counts.values,
    color=[LGREEN, "#FDD835", RED]
)

ax5.set_title(
    "Employee Risk Levels",
    fontsize=14,
    fontweight="bold"
)

ax5.set_ylabel("Count")

# ═══════════════════════════════════════════════════════════════
# SAVE DASHBOARD
# ═══════════════════════════════════════════════════════════════

plt.tight_layout()

plt.savefig(
    "charts/dashboard_attrition.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✅ Saved: charts/dashboard_attrition.png")

# ═══════════════════════════════════════════════════════════════
# ROC CURVES
# ═══════════════════════════════════════════════════════════════

# Prepare ML Data

df2 = pd.read_csv("data/hr_attrition.csv")

df2["Attrition_Binary"] = (
    df2["Attrition"] == "Yes"
).astype(int)

df2 = df2.drop(
    columns=["EmployeeNumber", "Attrition"],
    errors="ignore"
)

cat_cols = df2.select_dtypes(
    include="object"
).columns.tolist()

df2 = pd.get_dummies(
    df2,
    columns=cat_cols,
    drop_first=True
)

X = df2.drop(columns=["Attrition_Binary"])

y = df2["Attrition_Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

# Models

models = [

    (
        "Logistic Regression",

        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        X_train_sc,
        X_test_sc
    ),

    (
        "Random Forest",

        RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1
        ),

        X_train,
        X_test
    ),

    (
        "Gradient Boosting",

        GradientBoostingClassifier(
            n_estimators=50,
            random_state=42
        ),

        X_train,
        X_test
    )
]

fig2, ax = plt.subplots(
    figsize=(10, 7),
    facecolor=ACCENT
)

for name, model, Xtr, Xte in models:

    model.fit(Xtr, y_train)

    probs = model.predict_proba(Xte)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    auc = roc_auc_score(y_test, probs)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={auc:.3f})"
    )

ax.plot([0, 1], [0, 1], "k--")

ax.set_title(
    "ROC Curve Comparison",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.legend()

ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "charts/ml_results.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✅ Saved: charts/ml_results.png")

print("\n🎉 All Charts Generated Successfully")
print("Open charts/ folder")