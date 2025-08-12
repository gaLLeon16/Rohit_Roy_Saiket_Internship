import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")

data_path = r"C:\Users\rohit\OneDrive\Documents\Internship\Saiket\Rohit_Roy_Saiket_Internship\Task1\Telco_Customer_Churn_Dataset.csv"
df = pd.read_csv(data_path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] 

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features, y=top_features.index, palette='Blues_r')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.savefig("task5_feature_importance.png", bbox_inches='tight', dpi=300)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='navy')
plt.plot([0, 1], [0, 1], 'r--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("task5_roc_curve.png", bbox_inches='tight', dpi=300)
plt.show()

print(f" Task 5 complete. AUC Score: {roc_auc:.2f}")
print("Feature importance and ROC curve images saved.")
