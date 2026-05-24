"""
generate_data.py
Generates a realistic synthetic HR Employee Attrition dataset.
Run ONCE → creates data/hr_attrition.csv
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

N = 1470

DEPARTMENTS  = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES = {
    "Sales":                    ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development":   ["Research Scientist", "Laboratory Technician",
                                  "Healthcare Representative", "Manufacturing Director",
                                  "Research Director", "Manager"],
    "Human Resources":          ["Human Resources", "Manager"],
}
EDUCATION_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                    "Human Resources", "Other"]
GENDERS   = ["Male", "Female"]
MARITAL   = ["Single", "Married", "Divorced"]
OVERTIME   = ["Yes", "No"]
TRAVEL     = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]

rows = []
for i in range(N):
    dept   = random.choices(DEPARTMENTS, weights=[35, 55, 10])[0]
    role   = random.choice(JOB_ROLES[dept])
    gender = random.choice(GENDERS)
    age    = int(np.clip(np.random.normal(37, 9), 18, 60))
    years_at_company = int(np.clip(np.random.exponential(7), 0, 40))
    job_level        = random.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 12, 8])[0]
    monthly_income   = int(np.clip(
        np.random.normal(50000 + job_level * 20000, 15000), 15000, 200000))
    job_satisfaction       = random.choices([1,2,3,4], weights=[15,20,35,30])[0]
    env_satisfaction       = random.choices([1,2,3,4], weights=[10,20,35,35])[0]
    work_life_balance      = random.choices([1,2,3,4], weights=[10,15,45,30])[0]
    relationship_sat       = random.choices([1,2,3,4], weights=[10,20,35,35])[0]
    job_involvement        = random.choices([1,2,3,4], weights=[5,15,50,30])[0]
    overtime               = random.choices(OVERTIME, weights=[35, 65])[0]
    num_companies_worked   = random.randint(0, 9)
    training_times         = random.randint(0, 6)
    percent_salary_hike    = random.randint(11, 25)
    performance_rating     = random.choices([3,4], weights=[85, 15])[0]
    distance_from_home     = random.randint(1, 29)
    education              = random.randint(1, 5)
    education_field        = random.choice(EDUCATION_FIELDS)
    marital_status         = random.choice(MARITAL)
    business_travel        = random.choice(TRAVEL)
    stock_option           = random.choices([0,1,2,3], weights=[40,30,20,10])[0]
    years_in_role          = int(min(np.random.exponential(4), years_at_company))
    years_since_promotion  = int(min(np.random.exponential(2), years_at_company))
    years_with_manager     = int(min(np.random.exponential(4), years_at_company))

    # Attrition logic: higher probability for low satisfaction, overtime, low salary
    attrition_prob = 0.10
    if overtime == "Yes":             attrition_prob += 0.12
    if job_satisfaction <= 2:         attrition_prob += 0.10
    if work_life_balance <= 2:        attrition_prob += 0.08
    if monthly_income < 30000:        attrition_prob += 0.08
    if years_at_company <= 2:         attrition_prob += 0.07
    if num_companies_worked >= 5:     attrition_prob += 0.05
    if marital_status == "Single":    attrition_prob += 0.04
    if business_travel == "Travel_Frequently": attrition_prob += 0.05
    if env_satisfaction <= 2:         attrition_prob += 0.05
    attrition = "Yes" if random.random() < min(attrition_prob, 0.60) else "No"

    rows.append({
        "EmployeeNumber":       i + 1,
        "Age":                  age,
        "Attrition":            attrition,
        "BusinessTravel":       business_travel,
        "Department":           dept,
        "DistanceFromHome":     distance_from_home,
        "Education":            education,
        "EducationField":       education_field,
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender":               gender,
        "JobInvolvement":       job_involvement,
        "JobLevel":             job_level,
        "JobRole":              role,
        "JobSatisfaction":      job_satisfaction,
        "MaritalStatus":        marital_status,
        "MonthlyIncome":        monthly_income,
        "NumCompaniesWorked":   num_companies_worked,
        "OverTime":             overtime,
        "PercentSalaryHike":    percent_salary_hike,
        "PerformanceRating":    performance_rating,
        "RelationshipSatisfaction": relationship_sat,
        "StockOptionLevel":     stock_option,
        "TotalWorkingYears":    age - 18,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance":      work_life_balance,
        "YearsAtCompany":       years_at_company,
        "YearsInCurrentRole":   years_in_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
    })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/hr_attrition.csv", index=False)
print(f"✅ Dataset created: data/hr_attrition.csv  ({len(df):,} rows)")
print(f"   Attrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']:.1%}")
print(df.head(3))