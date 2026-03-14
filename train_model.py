import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# CSV load karo
df = pd.read_csv('student_expense_dataset.csv')

print("📊 Dataset Info:")
print(f"Total records: {len(df)}")
print(f"Categories: {df['category'].unique()}")
print(f"AI Advices: {df['spending_advice'].unique()}")
print()

# Prepare data
df_train = df.dropna(subset=['spending_advice']).copy()

le_cat = LabelEncoder()
le_subcat = LabelEncoder()
le_advice = LabelEncoder()

df_train["cat_encoded"] = le_cat.fit_transform(df_train["category"])
df_train["subcat_encoded"] = le_subcat.fit_transform(df_train["subcategory"])
df_train["advice_encoded"] = le_advice.fit_transform(df_train["spending_advice"])

X = df_train[["amount", "cat_encoded", "subcat_encoded"]]
y = df_train["advice_encoded"]

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔄 Train/Test Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()

# Train model
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("📈 Model Performance:")
print(f"✅ Training Accuracy: {train_accuracy:.2%}")
print(f"✅ Testing Accuracy: {test_accuracy:.2%}")
print()

# Classification report
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_advice.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🎯 Confusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_advice.classes_, yticklabels=le_advice.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n💾 Confusion matrix saved as confusion_matrix.png")

# Save models
os.makedirs('models', exist_ok=True)

with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/le_cat.pkl', 'wb') as f:
    pickle.dump(le_cat, f)

with open('models/le_subcat.pkl', 'wb') as f:
    pickle.dump(le_subcat, f)

with open('models/le_advice.pkl', 'wb') as f:
    pickle.dump(le_advice, f)

print("\n✅ Models saved!")
print(f"\n📋 Model Details:")
print(f"Max Depth: 5")
print(f"Feature importance: {dict(zip(['amount', 'category', 'subcategory'], model.feature_importances_))}")