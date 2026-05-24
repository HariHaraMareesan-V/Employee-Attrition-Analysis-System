
"""
02_ml_model.py
Full ML pipeline:
  - Feature engineering & encoding
  - Train Logistic Regression, Random Forest, Gradient Boosting
  - Evaluate metrics
  - Save feature importance & predictions
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("data", exist_ok=True)

print("=" * 65)
print("EMPLOYEE ATTRITION — MACHINE LEARNING PIPELINE")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════

df = pd.read_csv("data/hr_attrition.csv")

print(f"Dataset Shape: {df.shape}")

# Convert target to binary
df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)

# Drop unnecessary columns
drop_cols = ["EmployeeNumber", "Attrition"]

df = df.drop(columns=drop_cols, errors="ignore")

# ══════════════════════════════════════════════════════════════════════
# 2. ENCODING
# ══════════════════════════════════════════════════════════════════════

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\nCategorical Columns:")
print(cat_cols)

df = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

# Features & target
X = df.drop(columns=["Attrition_Binary"])
y = df["Attrition_Binary"]

feature_names = X.columns.tolist()

print(f"\nTotal Features: {len(feature_names)}")

# ══════════════════════════════════════════════════════════════════════
# 3. TRAIN TEST SPLIT
# ══════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTrain Size: {len(X_train)}")
print(f"Test Size : {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════
# 4. SCALING
# ══════════════════════════════════════════════════════════════════════

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════
# 5. MODELS
# ══════════════════════════════════════════════════════════════════════

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# ══════════════════════════════════════════════════════════════════════
# 6. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════

results = {}

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 65)
print("MODEL TRAINING & EVALUATION")
print("=" * 65)

for name, model in models.items():

    print(f"\n{'─'*55}")
    print(f"Model: {name}")
    print(f"{'─'*55}")

    # Logistic uses scaled data
    if name == "Logistic Regression":
        Xtr = X_train_sc
        Xte = X_test_sc
    else:
        Xtr = X_train
        Xte = X_test

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        Xtr,
        y_train,
        cv=skf,
        scoring="roc_auc",
        n_jobs=1
    )

    print(f"CV ROC-AUC: {cv_scores.mean():.4f}")

    # Train model
    model.fit(Xtr, y_train)

    # Predictions
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stay", "Leave"],
            zero_division=0
        )
    )

    # Save results
    results[name] = {

        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,

        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc
    }

# ══════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

comparison = pd.DataFrame([

    {
        "Model": name,
        "Accuracy": round(v["Accuracy"], 4),
        "Precision": round(v["Precision"], 4),
        "Recall": round(v["Recall"], 4),
        "F1": round(v["F1"], 4),
        "ROC-AUC": round(v["ROC-AUC"], 4)
    }

    for name, v in results.items()
])

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(comparison)

comparison.to_csv(
    "data/model_comparison.csv",
    index=False
)

# ══════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

rf_model = results["Random Forest"]["model"]

importances = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
)

top20 = (
    importances
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

top20.columns = ["Feature", "Importance"]

print("\nTop Feature Importances:")
print(top20)

top20.to_csv(
    "data/feature_importance.csv",
    index=False
)

# ══════════════════════════════════════════════════════════════════════
# 9. SAVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════

best_model = results["Random Forest"]

preds_df = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_model["y_pred"],

    "Attrition_Probability":
        np.round(best_model["y_prob"], 4)
})

preds_df.to_csv(
    "data/predictions.csv",
    index=False
)

print("\n✅ Saved: data/model_comparison.csv")
print("✅ Saved: data/feature_importance.csv")
print("✅ Saved: data/predictions.csv")

print("\n🎉 ML Pipeline Completed Successfully")
print("Run: python 03_visualizations.py")

