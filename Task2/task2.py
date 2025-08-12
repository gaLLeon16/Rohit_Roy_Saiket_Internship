import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

data_path = r"C:\Users\rohit\OneDrive\Documents\Internship\Saiket\Rohit_Roy_Saiket_Internship\Task1\Telco_Customer_Churn_Dataset.csv"
df = pd.read_csv(data_path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=["skyblue", "lightcoral"])
plt.title("Overall Churn Rate")
plt.savefig("output_1_churn_rate.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='gender', hue='Churn', palette='Set2')
plt.title("Churn by Gender")
plt.savefig("output_2_churn_by_gender.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Partner', hue='Churn', palette='Set1')
plt.title("Churn by Partner Status")
plt.savefig("output_3_churn_by_partner.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Dependents', hue='Churn', palette='Pastel1')
plt.title("Churn by Dependents")
plt.savefig("output_4_churn_by_dependents.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn', palette='coolwarm')
plt.title("Churn by Contract Type")
plt.savefig("output_5_churn_by_contract.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette='Set3')
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_6_churn_by_payment_method.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, palette='magma')
plt.title("Tenure vs Churn")
plt.savefig("output_7_tenure_vs_churn.png", bbox_inches="tight", dpi=300)
plt.show()
