# 🧑‍💼 Employee Attrition Analysis System

## Project Structure
```
project2/
├── generate_data.py       ← Step 0: Generate synthetic HR dataset
├── 01_eda_sql.py          ← Step 1: EDA + 7 SQL queries via SQLite
├── 02_ml_model.py         ← Step 2: Train & evaluate 3 ML models
├── 03_visualizations.py   ← Step 3: Full dashboard + ML result charts
├── data/                  ← Auto-created CSVs + SQLite DB
└── charts/                ← Auto-created PNGs


## What You Get
- 7 SQL queries (attrition by dept, salary bands, tenure, overtime, etc.)
- 3 trained classifiers: Logistic Regression, Random Forest, Gradient Boosting
- Full evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Feature Importance chart (top 15 drivers of attrition)
- Risk scoring per employee (Low / Medium / High)
- ROC curves, probability distribution plots

## Key Findings (expected)
- Employees with OverTime=Yes have ~2x higher attrition rate
- Low income + low job satisfaction = highest attrition risk
- 0-2 year tenure employees leave most frequently
- Single employees leave more than married ones
- Random Forest achieves ~85%+ ROC-AUC
