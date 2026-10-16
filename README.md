# SpendSmart: Personal Expense Tracker

A Streamlit app that helps students track spending, see where their money goes, and plan toward a savings goal. A small machine-learning model labels each expense with spending advice ("Reduce", "Acceptable", …).

## Features

| Tab | What it does |
|---|---|
| **Overview** | Total spend, spending by category, daily spending trend, category and sub-category breakdowns (Plotly) |
| **Add expense** | Log a new expense; the model returns spending advice for it |
| **Savings Goal Engine** | Enter income, goal amount and target month to get the monthly saving required and the gap to close |
| **Upload & report** | Upload your own expense CSV and generate a detailed report |

**ML:** a Decision Tree classifier (scikit-learn) on `amount`, `category` and `subcategory` predicts the advice label. `train_model.py` uses a stratified train/test split, prints a classification report and saves a confusion matrix.

> The bundled `student_expense_dataset.csv` is a small sample (78 rows), so model scores on it are illustrative only.

## Versions in this repo

| File | Storage | Notes |
|---|---|---|
| `app.py` | Local CSV | Simplest version, no setup needed |
| `app_firebase.py` | Firebase Firestore | Expenses stored in the cloud |
| `app_with_auth_v2.py` | Firebase Firestore | Adds register/login (SHA-256 hashed passwords) and a per-user model that retrains on the user's own data |
| `train_model.py` | – | Trains and evaluates the baseline model and saves it to `models/` |
| `Savings_Goal_Engine.py`, `main.py` | – | Command-line prototypes of the goal planner and the classifier |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the Firebase versions, create a Firebase project and download a service-account key as `serviceAccountKey.json` into this folder. **Never commit that file.** It is listed in `.gitignore`.

## Tech

Python · Pandas · scikit-learn · Streamlit · Plotly · Firebase Admin SDK

## Author

**Raghav Dhir**: [LinkedIn](https://www.linkedin.com/in/raghavdhirr)
