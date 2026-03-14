import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv('C:\\Users\\HP\\Desktop\\expense_app\\student_expense_dataset.csv')

# Encode categorical columns
le_category = LabelEncoder()
le_subcategory = LabelEncoder()
le_advice = LabelEncoder()

df["category"] = le_category.fit_transform(df["category"])
df["subcategory"] = le_subcategory.fit_transform(df["subcategory"])
df["spending_advice"] = le_advice.fit_transform(df["spending_advice"])

X = df[["amount","category","subcategory"]]
y = df["spending_advice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))