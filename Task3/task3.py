import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

data_path = r"C:\Users\rohit\OneDrive\Documents\Internship\Saiket\Rohit_Roy_Saiket_Internship\Task1\Telco_Customer_Churn_Dataset.csv"
df = pd.read_csv(data_path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['TenureGroup'] = pd.cut(df['tenure'],
                           bins=[0, 12, 24, 48, 72],
                           labels=['0–12', '13–24', '25–48', '49–72'])

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='TenureGroup', hue='Churn', palette='Set2')
plt.title("Churn by Tenure Group")
plt.savefig("segmentation_1_tenure.png", bbox_inches='tight', dpi=300)
plt.show()

df['ChargeGroup'] = pd.cut(df['MonthlyCharges'],
                           bins=[0, 35, 70, 100, 150],
                           labels=['Low', 'Medium', 'High', 'Very High'])

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='ChargeGroup', hue='Churn', palette='coolwarm')
plt.title("Churn by Monthly Charges")
plt.savefig("segmentation_2_monthly_charges.png", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
plt.title("Churn by Contract Type")
plt.savefig("segmentation_3_contract.png", bbox_inches='tight', dpi=300)
plt.show()

high_risk = df[(df['ChargeGroup'].isin(['High', 'Very High'])) &
               (df['TenureGroup'].isin(['0–12', '13–24'])) &
               (df['Churn'] == 1)]

print(" Top High-Value, High-Risk Customers:")
print(high_risk[['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure', 'Contract']].head())

high_risk[['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure', 'Contract']].to_csv("high_risk_customers.csv", index=False)
