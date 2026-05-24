"""
01_eda_sql.py
Exploratory Data Analysis + SQL-based querying using sqlite3.
"""

import pandas as pd
import numpy as np
import sqlite3
import os

os.makedirs("data", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/hr_attrition.csv")
print("=" * 65)
print("EMPLOYEE ATTRITION — EDA & SQL ANALYSIS")
print("=" * 65)
print(f"Shape: {df.shape}   |   Columns: {df.shape[1]}")
print(f"\nAttrition Distribution:\n{df['Attrition'].value_counts()}")
attrition_rate = (df['Attrition'] == 'Yes').mean()
print(f"\nOverall Attrition Rate: {attrition_rate:.1%}")

# ── Basic cleaning ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("DATA QUALITY CHECK")
print("=" * 65)
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"\nData types:\n{df.dtypes}")

# ══════════════════════════════════════════════════════════════════════════════
#  SQL ANALYSIS — Load into SQLite
# ══════════════════════════════════════════════════════════════════════════════
conn = sqlite3.connect("data/hr_analysis.db")
df.to_sql("hr_data", conn, if_exists="replace", index=False)
print("\n✅ Data loaded into SQLite (data/hr_analysis.db)")

def run_query(title, sql):
    print(f"\n{'─'*65}")
    print(f"  SQL: {title}")
    print(f"{'─'*65}")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    return result

# SQL 1: Attrition rate by Department
dept_attrition = run_query(
    "Attrition Rate by Department",
    """
    SELECT
        Department,
        COUNT(*) AS Total_Employees,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2)
            AS Attrition_Rate_Pct
    FROM hr_data
    GROUP BY Department
    ORDER BY Attrition_Rate_Pct DESC
    """
)
dept_attrition.to_csv("data/sql_dept_attrition.csv", index=False)

# SQL 2: Avg salary — leavers vs stayers
run_query(
    "Avg Salary: Leavers vs Stayers",
    """
    SELECT
        Attrition,
        ROUND(AVG(MonthlyIncome), 2)    AS Avg_Monthly_Income,
        ROUND(AVG(YearsAtCompany), 2)   AS Avg_Tenure_Years,
        ROUND(AVG(JobSatisfaction), 2)  AS Avg_Job_Satisfaction,
        ROUND(AVG(WorkLifeBalance), 2)  AS Avg_WLB,
        COUNT(*) AS Count
    FROM hr_data
    GROUP BY Attrition
    """
)

# SQL 3: Attrition by OverTime
overtime_attr = run_query(
    "Attrition by OverTime",
    """
    SELECT
        OverTime,
        COUNT(*) AS Total,
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2)
            AS Attrition_Pct
    FROM hr_data
    GROUP BY OverTime
    """
)
overtime_attr.to_csv("data/sql_overtime_attrition.csv", index=False)

# SQL 4: Attrition by JobRole
run_query(
    "Top Job Roles by Attrition Rate",
    """
    SELECT
        JobRole,
        COUNT(*) AS Total,
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2)
            AS Attrition_Pct
    FROM hr_data
    GROUP BY JobRole
    HAVING Total >= 30
    ORDER BY Attrition_Pct DESC
    """
)

# SQL 5: Income band analysis
run_query(
    "Attrition by Monthly Income Band",
    """
    SELECT
        CASE
            WHEN MonthlyIncome < 25000  THEN 'Low (<25K)'
            WHEN MonthlyIncome < 50000  THEN 'Mid (25K-50K)'
            WHEN MonthlyIncome < 100000 THEN 'High (50K-1L)'
            ELSE 'Very High (>1L)'
        END AS Income_Band,
        COUNT(*) AS Total,
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2)
            AS Attrition_Pct
    FROM hr_data
    GROUP BY Income_Band
    ORDER BY Attrition_Pct DESC
    """
)

# SQL 6: Marital status impact
run_query(
    "Attrition by Marital Status",
    """
    SELECT
        MaritalStatus,
        COUNT(*) AS Total,
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2)
            AS Attrition_Pct
    FROM hr_data
    GROUP BY MaritalStatus
    ORDER BY Attrition_Pct DESC
    """
)

# SQL 7: Years at company vs attrition
run_query(
    "Attrition by Tenure Band",
    """
    SELECT
        CASE
            WHEN YearsAtCompany <= 2  THEN '0-2 Years'
            WHEN YearsAtCompany <= 5  THEN '3-5 Years'
            WHEN YearsAtCompany <= 10 THEN '6-10 Years'
            ELSE '10+ Years'
        END AS Tenure_Band,
        COUNT(*) AS Total,
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Left_Count,
        ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2)
            AS Attrition_Pct
    FROM hr_data
    GROUP BY Tenure_Band
    ORDER BY Attrition_Pct DESC
    """
)

conn.close()
print("\n✅ All SQL queries complete. CSVs saved to data/")
print("   Run 02_ml_model.py next.")