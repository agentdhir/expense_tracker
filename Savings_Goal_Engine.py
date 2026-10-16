import pandas as pd
from datetime import datetime

df = pd.read_csv("student_expense_dataset.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# User inputs
monthly_income = 2000
saving_goal = 3000
goal_target_month = 6
goal_type = "Laptop"

current_month = datetime.now().month

# Average monthly spend
monthly_spend = df["amount"].sum() / df["date"].dt.month.nunique()

# Months left (a target month already passed this year rolls over to next year)
months_left = (goal_target_month - current_month) % 12 or 12

required_monthly_saving = saving_goal / months_left

current_saving = monthly_income - monthly_spend

saving_gap = required_monthly_saving - current_saving

print("Goal:", goal_type)
print("Target saving:", saving_goal)
print("Months left:", months_left)

print("\nRequired monthly saving:", round(required_monthly_saving,2))
print("Current saving capacity:", round(current_saving,2))

if saving_gap > 0:
    print("You must reduce spending by:", round(saving_gap,2))
else:
    print("Goal achievable without reducing spending")