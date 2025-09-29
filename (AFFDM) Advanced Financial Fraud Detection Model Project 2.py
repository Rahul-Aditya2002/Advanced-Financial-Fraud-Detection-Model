#!/usr/bin/env python
# coding: utf-8

# # INFOTACT SOLUTION PROJECT 2

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
df_credit = pd.read_csv("creditcard.csv")
df_credit.head()


# In[3]:


df_financial = pd.read_csv("Financial_Fraud_detection.csv")
df_financial.head()


# In[4]:


df_paysim = pd.read_csv("Synthetic_Financial_datasets_log.csv")
df_paysim.head()


# In[5]:


df_credit.info()
df_financial.info()
df_paysim.info()


# In[6]:


# Checking for missing values in all datasets
print("Missing values in Credit Card Dataset:")
print(df_credit.isnull().sum(), "\n")

print("Missing values in Financial Fraud Detection Dataset:")
print(df_financial.isnull().sum(), "\n")

print("Missing values in PaySim Dataset:")
print(df_paysim.isnull().sum(), "\n")


# In[7]:


# Percentage of missing values in each column
print("Credit Card Dataset:")
print(df_credit.isnull().sum() / len(df_credit) * 100, "\n")

print("Financial Fraud Detection Dataset:")
print(df_financial.isnull().sum() / len(df_financial) * 100, "\n")

print("PaySim Dataset:")
print(df_paysim.isnull().sum() / len(df_paysim) * 100, "\n")


# In[7]:


print("Credit Card Dataset Duplicates:", df_credit.duplicated().sum())
print("Financial Fraud Detection Dataset Duplicates:", df_financial.duplicated().sum())
print("PaySim Dataset Duplicates:", df_paysim.duplicated().sum())


# In[8]:


df_credit = df_credit.drop_duplicates()
print("Duplicates removed. New shape:", df_credit.shape)


# In[9]:


# Checking for negative transaction amounts
print("Negative transaction amounts in Credit Card dataset:", (df_credit['Amount'] < 0).sum())
print("Negative transaction amounts in Financial Fraud dataset:", (df_financial['amount'] < 0).sum())
print("Negative transaction amounts in PaySim dataset:", (df_paysim['amount'] < 0).sum())


# In[10]:


# Checking for balance inconsistencies in Financial and PaySim datasets
df_financial_invalid = df_financial[(df_financial['oldbalanceOrg'] < 0) | (df_financial['newbalanceOrig'] < 0) |
                                    (df_financial['oldbalanceDest'] < 0) | (df_financial['newbalanceDest'] < 0)]

df_paysim_invalid = df_paysim[(df_paysim['oldbalanceOrg'] < 0) | (df_paysim['newbalanceOrig'] < 0) |
                              (df_paysim['oldbalanceDest'] < 0) | (df_paysim['newbalanceDest'] < 0)]

print("Invalid balance values in Financial Fraud dataset:", df_financial_invalid.shape[0])
print("Invalid balance values in PaySim dataset:", df_paysim_invalid.shape[0])


# In[11]:


# Checking unique values in the 'type' column
print("Unique transaction types in Financial Fraud dataset:", df_financial['type'].unique())
print("Unique transaction types in PaySim dataset:", df_paysim['type'].unique())


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Setting figure size
plt.figure(figsize=(15, 5))

# Credit Card Fraud Dataset
plt.subplot(1, 3, 1)
sns.boxplot(x=df_credit['Amount'])
plt.title("Credit Card Fraud - Transaction Amounts")

# Financial Fraud Detection Dataset
plt.subplot(1, 3, 2)
sns.boxplot(x=df_financial['amount'])
plt.title("Financial Fraud Detection - Transaction Amounts")

# PaySim Fraud Dataset
plt.subplot(1, 3, 3)
sns.boxplot(x=df_paysim['amount'])
plt.title("PaySim - Transaction Amounts")

plt.tight_layout()
plt.show()


# In[13]:


# Checking fraud labels in outlier transactions
high_amount_threshold = {
    "credit": df_credit["Amount"].quantile(0.99),
    "financial": df_financial["amount"].quantile(0.99),
    "paysim": df_paysim["amount"].quantile(0.99),
}

# Filtering high-value transactions
outliers_credit = df_credit[df_credit["Amount"] > high_amount_threshold["credit"]]
outliers_financial = df_financial[df_financial["amount"] > high_amount_threshold["financial"]]
outliers_paysim = df_paysim[df_paysim["amount"] > high_amount_threshold["paysim"]]

# Checking fraud percentages
fraud_credit = outliers_credit["Class"].mean() * 100
fraud_financial = outliers_financial["isFraud"].mean() * 100
fraud_paysim = outliers_paysim["isFraud"].mean() * 100

print(f"Fraud percentage in high-value transactions:")
print(f"Credit Card Dataset: {fraud_credit:.2f}%")
print(f"Financial Fraud Dataset: {fraud_financial:.2f}%")
print(f"PaySim Dataset: {fraud_paysim:.2f}%")


# In[14]:


# Removing outliers based on the 99th percentile
df_credit = df_credit[df_credit["Amount"] <= high_amount_threshold["credit"]]
df_financial = df_financial[df_financial["amount"] <= high_amount_threshold["financial"]]
df_paysim = df_paysim[df_paysim["amount"] <= high_amount_threshold["paysim"]]

# Checking new shapes after outlier removal
print(f"New shape of Credit Card Dataset: {df_credit.shape}")
print(f"New shape of Financial Fraud Dataset: {df_financial.shape}")
print(f"New shape of PaySim Dataset: {df_paysim.shape}")


# In[15]:


# Plot transaction type distribution
plt.figure(figsize=(12, 5))

# Financial Fraud Dataset
plt.subplot(1, 2, 1)
sns.countplot(x=df_financial['type'], order=df_financial['type'].value_counts().index, palette="viridis")
plt.title("Financial Fraud Dataset - Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=45)


# In[16]:


# PaySim Dataset
plt.subplot(1, 2, 2)
sns.countplot(x=df_paysim['type'], order=df_paysim['type'].value_counts().index, palette="magma")
plt.title("PaySim Dataset - Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[21]:


# Fraud vs. Non-Fraud Transactions
plt.figure(figsize=(12, 5))

# Credit Card Dataset
plt.subplot(1, 3, 1)
sns.countplot(x=df_credit['Class'], palette="coolwarm")
plt.title("Credit Card Dataset - Fraud vs. Non-Fraud")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")

# Financial Fraud Dataset
plt.subplot(1, 3, 2)
sns.countplot(x=df_financial['isFraud'], palette="coolwarm")
plt.title("Financial Fraud Dataset - Fraud vs. Non-Fraud")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")

# PaySim Dataset
plt.subplot(1, 3, 3)
sns.countplot(x=df_paysim['isFraud'], palette="coolwarm")
plt.title("PaySim Dataset - Fraud vs. Non-Fraud")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# In[24]:


# Distribution of Transaction Amounts for Fraud & Non-Fraud Transactions
plt.figure(figsize=(12, 5))

# Credit Card Dataset
plt.subplot(1, 3, 1)
sns.boxplot(x=df_credit['Class'], y=df_credit['Amount'], palette="coolwarm")
plt.title("Credit Card Dataset - Transaction Amounts")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Amount")

# Financial Fraud Dataset
plt.subplot(1, 3, 2)
sns.boxplot(x=df_financial['isFraud'], y=df_financial['amount'], palette="coolwarm")
plt.title("Financial Fraud Dataset - Transaction Amounts")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Amount")

# PaySim Dataset
plt.subplot(1, 3, 3)
sns.boxplot(x=df_paysim['isFraud'], y=df_paysim['amount'], palette="coolwarm")
plt.title("PaySim Dataset - Transaction Amounts")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df_credit["Amount"])
plt.title("Credit Card - Transaction Amounts")

plt.subplot(1, 3, 2)
sns.boxplot(y=df_financial["amount"])
plt.title("Financial Fraud - Transaction Amounts")

plt.subplot(1, 3, 3)
sns.boxplot(y=df_paysim["amount"])
plt.title("PaySim - Transaction Amounts")

plt.show()


# In[26]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df_credit["Class"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightblue", "red"], labels=["Non-Fraud", "Fraud"])
plt.title("Credit Card Fraud Percentage")

plt.subplot(1, 3, 2)
df_financial["isFraud"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightblue", "red"], labels=["Non-Fraud", "Fraud"])
plt.title("Financial Fraud Percentage")

plt.subplot(1, 3, 3)
df_paysim["isFraud"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightblue", "red"], labels=["Non-Fraud", "Fraud"])
plt.title("PaySim Fraud Percentage")

plt.show()


# In[27]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_credit["Amount"], bins=50, kde=True, color="blue")
plt.title("Credit Card - Transaction Amounts")

plt.subplot(1, 3, 2)
sns.histplot(df_financial["amount"], bins=50, kde=True, color="green")
plt.title("Financial Fraud - Transaction Amounts")

plt.subplot(1, 3, 3)
sns.histplot(df_paysim["amount"], bins=50, kde=True, color="purple")
plt.title("PaySim - Transaction Amounts")

plt.show()


# In[28]:


plt.figure(figsize=(10, 6))
sns.heatmap(df_credit.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Credit Card Dataset - Feature Correlations")
plt.show()


# In[29]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.violinplot(x=df_credit["Class"], y=df_credit["Amount"], palette=["lightblue", "red"])
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.title("Credit Card - Fraud vs. Non-Fraud Amounts")

plt.subplot(1, 3, 2)
sns.violinplot(x=df_financial["isFraud"], y=df_financial["amount"], palette=["lightblue", "red"])
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.title("Financial Fraud - Fraud vs. Non-Fraud Amounts")

plt.subplot(1, 3, 3)
sns.violinplot(x=df_paysim["isFraud"], y=df_paysim["amount"], palette=["lightblue", "red"])
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.title("PaySim - Fraud vs. Non-Fraud Amounts")

plt.show()


# In[30]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.lineplot(x=df_credit.index, y=df_credit["Amount"], color="blue")
plt.title("Credit Card - Transaction Trends")

plt.subplot(1, 3, 2)
sns.lineplot(x=df_financial["step"], y=df_financial["amount"], color="green")
plt.title("Financial Fraud - Transaction Trends")

plt.subplot(1, 3, 3)
sns.lineplot(x=df_paysim["step"], y=df_paysim["amount"], color="purple")
plt.title("PaySim - Transaction Trends")

plt.show()


# In[31]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df_financial["oldbalanceOrg"], y=df_financial["amount"], hue=df_financial["isFraud"], palette=["blue", "red"])
plt.title("Financial Fraud - Amount vs. Old Balance")

plt.subplot(1, 2, 2)
sns.scatterplot(x=df_paysim["oldbalanceOrg"], y=df_paysim["amount"], hue=df_paysim["isFraud"], palette=["blue", "red"])
plt.title("PaySim - Amount vs. Old Balance")

plt.show()


# # Machine Learning

# In[17]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df_financial["type"] = encoder.fit_transform(df_financial["type"])
df_paysim["type"] = encoder.transform(df_paysim["type"])  # Use same encoder


# In[18]:


from sklearn.model_selection import train_test_split

X_credit = df_credit.drop(columns=["Class"])
y_credit = df_credit["Class"]

X_financial = df_financial.drop(columns=["isFraud"])
y_financial = df_financial["isFraud"]

X_paysim = df_paysim.drop(columns=["isFraud"])
y_paysim = df_paysim["isFraud"]

# Splitting into 80% Train & 20% Test
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
X_train_financial, X_test_financial, y_train_financial, y_test_financial = train_test_split(X_financial, y_financial, test_size=0.2, random_state=42)
X_train_paysim, X_test_paysim, y_train_paysim, y_test_paysim = train_test_split(X_paysim, y_paysim, test_size=0.2, random_state=42)


# In[19]:


from sklearn.preprocessing import StandardScaler

non_numeric_cols = X_train_financial.select_dtypes(exclude=["number"]).columns

# Drop non-numeric columns
X_train_financial = X_train_financial.drop(columns=non_numeric_cols)
X_test_financial = X_test_financial.drop(columns=non_numeric_cols)

X_train_paysim = X_train_paysim.drop(columns=non_numeric_cols)
X_test_paysim = X_test_paysim.drop(columns=non_numeric_cols)

# Now apply StandardScaler
scaler = StandardScaler()

X_train_financial = scaler.fit_transform(X_train_financial)
X_test_financial = scaler.transform(X_test_financial)

X_train_paysim = scaler.fit_transform(X_train_paysim)
X_test_paysim = scaler.transform(X_test_paysim)


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize model
log_reg = LogisticRegression()

# Train on Financial Fraud dataset
log_reg.fit(X_train_financial, y_train_financial)

# Predictions
y_pred_financial = log_reg.predict(X_test_financial)

# Evaluation
print("Logistic Regression - Financial Fraud Dataset")
print("Accuracy:", accuracy_score(y_test_financial, y_pred_financial))
print(classification_report(y_test_financial, y_pred_financial))


# In[22]:


from sklearn.ensemble import RandomForestClassifier

# Initialize model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on Financial Fraud dataset
rf_clf.fit(X_train_financial, y_train_financial)

# Predictions
y_pred_rf_financial = rf_clf.predict(X_test_financial)

# Evaluation
print("Random Forest - Financial Fraud Dataset")
print("Accuracy:", accuracy_score(y_test_financial, y_pred_rf_financial))
print(classification_report(y_test_financial, y_pred_rf_financial))


# In[23]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test_financial, y_pred_financial)

# Display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# # Using Machine Algorithms

# In[40]:


from sklearn.linear_model import LogisticRegression

# Initialize the model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train_financial, y_train_financial)


# In[41]:


y_pred_prob_financial = logistic_model.predict_proba(X_test_financial)[:, 1]  # Get probabilities


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(y_pred_prob_financial, bins=50, kde=True, color="red")
plt.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Frequency")
plt.title("Fraud vs. Non-Fraud Distribution")
plt.legend()
plt.show()


# In[108]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test_financial, y_pred_prob_financial)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[111]:


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_financial, y_pred_prob_financial)

# Plot the curve
plt.figure()
plt.plot(recall, precision, marker=".", color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# In[ ]:


# # Export cleaned datasets to CSV
# df_credit.to_csv("Cleaned_Credit_Card.csv", index=False)
# df_financial.to_csv("Cleaned_Financial_Fraud.csv", index=False)
# df_paysim.to_csv("Cleaned_PaySim.csv", index=False)

# print("Export completed: Cleaned_Credit_Card.csv, Cleaned_Financial_Fraud.csv, Cleaned_PaySim.csv")


# # User Behavior Pattern Analysis

# In[43]:


# Use 'nameOrig' as the Customer ID
df_financial['customer_id'] = df_financial['nameOrig']

# Use 'step' as a proxy for time-based analysis
df_financial['transaction_hour'] = df_financial['step'] % 24  # Assuming 'step' is in hours
df_financial['transaction_day'] = df_financial['step'] // 24  # Converting steps into days

# Aggregating user behavior statistics
user_behavior = df_financial.groupby('customer_id').agg({
    'amount': ['mean', 'max', 'count'],
    'transaction_hour': ['mean'],
    'transaction_day': ['mean']
}).reset_index()

# Rename columns for better readability
user_behavior.columns = ['customer_id', 'avg_transaction_amt', 'max_transaction_amt', 'total_transactions', 
                         'avg_transaction_hour', 'avg_transaction_day']

# Display results
print(user_behavior.head())


# In[31]:


# Merge fraud labels with user behavior data
df_financial['customer_id'] = df_financial['nameOrig']
user_behavior = user_behavior.merge(df_financial[['customer_id', 'isFraud']], on='customer_id', how='left')

# Analyzing Fraudulent vs. Non-Fraudulent Transactions
fraud_users = user_behavior[user_behavior['isFraud'] == 1]
non_fraud_users = user_behavior[user_behavior['isFraud'] == 0]

# Compare transaction frequency
plt.figure(figsize=(12, 5))
sns.kdeplot(non_fraud_users['total_transactions'], label="Non-Fraud", shade=True, color="green")
sns.kdeplot(fraud_users['total_transactions'], label="Fraud", shade=True, color="red")
plt.xlabel("Total Transactions per User")
plt.ylabel("Density")
plt.title("Fraud vs. Non-Fraud Transaction Frequency")
plt.legend()
plt.show()


# In[32]:


plt.figure(figsize=(12, 5))
sns.kdeplot(non_fraud_users['avg_transaction_hour'], label="Non-Fraud", shade=True, color="blue")
sns.kdeplot(fraud_users['avg_transaction_hour'], label="Fraud", shade=True, color="red")
plt.xlabel("Average Transaction Hour")
plt.ylabel("Density")
plt.title("Fraud vs. Non-Fraud Transaction Timing Patterns")
plt.legend()
plt.show()


# In[33]:


plt.figure(figsize=(12, 5))
sns.kdeplot(non_fraud_users['avg_transaction_amt'], label="Non-Fraud", shade=True, color="purple")
sns.kdeplot(fraud_users['avg_transaction_amt'], label="Fraud", shade=True, color="orange")
plt.xlabel("Average Transaction Amount")
plt.ylabel("Density")
plt.title("Fraud vs. Non-Fraud Transaction Amount Distribution")
plt.legend()
plt.show()


# In[37]:


# Compare Heuristic-Based Detection with Actual Fraud Labels

# Ensure customer_id is unique before mapping
user_behavior_unique = user_behavior.groupby('customer_id').agg({
    'avg_transaction_amt': 'mean',
    'max_transaction_amt': 'mean',
    'total_transactions': 'mean',
    'avg_transaction_hour': 'mean',
    'avg_transaction_day': 'mean'
}).reset_index()

# Define thresholds based on previous analysis
UNUSUAL_HOUR_THRESHOLD = [0, 1, 2, 3, 4]  # Suspicious hours (Midnight to 4 AM)
HIGH_TRANSACTION_FREQUENCY = user_behavior_unique['total_transactions'].quantile(0.95)  # Top 5% frequency
HIGH_TRANSACTION_AMOUNT = user_behavior_unique['avg_transaction_amt'].quantile(0.95)  # Top 5% transaction amount

# Create fraud suspicion flag
df_financial['fraud_suspicion'] = 0  # Default: No suspicion

# Rule 1: Unusual Transaction Timing
df_financial.loc[df_financial['transaction_hour'].isin(UNUSUAL_HOUR_THRESHOLD), 'fraud_suspicion'] = 1

# Rule 2: High Transaction Frequency
df_financial.loc[df_financial['customer_id'].map(user_behavior_unique.set_index('customer_id')['total_transactions']) > HIGH_TRANSACTION_FREQUENCY, 'fraud_suspicion'] = 1

# Rule 3: Unusually Large Transactions
df_financial.loc[df_financial['customer_id'].map(user_behavior_unique.set_index('customer_id')['avg_transaction_amt']) > HIGH_TRANSACTION_AMOUNT, 'fraud_suspicion'] = 1

# Display flagged suspicious transactions
df_financial[df_financial['fraud_suspicion'] == 1].head()


# In[38]:


# Evaluate the Heuristics

# Confusion matrix for heuristic fraud detection
from sklearn.metrics import confusion_matrix

y_true = df_financial['isFraud']  # Actual fraud labels
y_pred = df_financial['fraud_suspicion']  # Our heuristic predictions

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix: Heuristic-Based Fraud Detection")
plt.show()


# # Historical Data Utilization – Fraud Trend Analysis

# In[39]:


# Convert 'step' to days (each step is 1 hour, so divide by 24)
df_paysim['days'] = df_paysim['step'] // 24

# Aggregate fraud cases per day
fraud_trend = df_paysim[df_paysim['isFraud'] == 1].groupby('days').size().reset_index(name='fraud_count')

# Display data sample
fraud_trend.head()


# In[40]:


plt.figure(figsize=(12, 5))
sns.lineplot(x='days', y='fraud_count', data=fraud_trend, marker='o', color='red')
plt.title("Fraud Cases Over Time")
plt.xlabel("Days")
plt.ylabel("Number of Fraud Cases")
plt.grid(True)
plt.show()


# In[42]:


# Aggregate User-Level Historical Statistics

# Aggregate historical fraud patterns per customer
user_history = df_paysim.groupby('nameOrig').agg({
    'isFraud': ['sum', 'count'],  # Total fraud cases & total transactions per user
    'amount': ['mean', 'max'],    # Avg & max transaction amount per user
}).reset_index()

# Rename columns for better readability
user_history.columns = ['customer_id', 'past_fraud_count', 'total_transactions', 'avg_transaction_amt', 'max_transaction_amt']

# Compute fraud rate per user
user_history['fraud_rate'] = user_history['past_fraud_count'] / user_history['total_transactions']

# Display sample
user_history.head()


# In[43]:


# Merge past fraud insights into the original dataset
df_paysim = df_paysim.merge(user_history, how='left', left_on='nameOrig', right_on='customer_id')

# Fill missing values (new customers with no fraud history)
df_paysim[['past_fraud_count', 'fraud_rate']] = df_paysim[['past_fraud_count', 'fraud_rate']].fillna(0)

# Display updated data
df_paysim.head()


# In[47]:


# Define features (including historical fraud patterns)

# from sklearn.ensemble import RandomForestClassifier

features = ['amount', 'fraud_rate', 'past_fraud_count', 'avg_transaction_amt', 'max_transaction_amt']
X = df_paysim[features]
y = df_paysim['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Model performance
print(classification_report(y_test, y_pred))


# # Scalability and Customization

# In[58]:


# Define fraud detection thresholds
HIGH_TRANSACTION_AMOUNT = 10000  
FREQUENT_TRANSACTIONS = 5  

# Apply fraud detection using vectorized operations (FAST)
df_financial["fraud_rule_1"] = df_financial["amount"] > HIGH_TRANSACTION_AMOUNT
df_financial["fraud_rule_2"] = df_financial["isFraud"] == 1
df_financial["final_fraud_decision"] = df_financial[["fraud_rule_1", "fraud_rule_2"]].any(axis=1)

# Display a sample of detected fraud cases
df_financial[df_financial["final_fraud_decision"]].head()


# In[59]:


def detect_fraud(amount, isFraud, threshold=10000):
    return (amount > threshold) or (isFraud == 1)

df_financial["fraud_detected"] = df_financial.apply(lambda x: detect_fraud(x["amount"], x["isFraud"]), axis=1)


# In[60]:


df_financial.to_csv("financial_fraud_results.csv", index=False)
print("Fraud detection results saved successfully!")


# In[61]:


# Define fraud detection thresholds
HIGH_TRANSACTION_AMOUNT = 10000  

# Apply fraud detection using vectorized operations
df_financial["fraud_rule_1"] = df_financial["amount"] > HIGH_TRANSACTION_AMOUNT
df_financial["fraud_rule_2"] = df_financial["isFraud"] == 1

# Combine fraud detection rules
df_financial["final_fraud_decision"] = df_financial[["fraud_rule_1", "fraud_rule_2"]].any(axis=1)

# Display first 5 detected fraud cases
df_financial[df_financial["final_fraud_decision"]].head()


# In[62]:


# Function to detect fraud based on customizable rules
def detect_fraud(amount, isFraud, threshold=10000):
    return (amount > threshold) or (isFraud == 1)

# Apply function to dataset
df_financial["fraud_detected"] = df_financial.apply(lambda x: detect_fraud(x["amount"], x["isFraud"]), axis=1)


# In[63]:


df_financial.to_csv("financial_fraud_results.csv", index=False)
print("Fraud detection results saved successfully!")


# In[ ]:





# In[ ]:





# In[ ]:




