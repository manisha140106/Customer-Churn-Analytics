from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Segmentation & Churn Analytics")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('European_Bank.csv')
    
    # Data Cleaning
    df.drop(['Surname'], axis=1, inplace=True)
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    
    # Feature Engineering
    
    # Age Group
    def age_group(age):
        if age < 30:
            return "<30"
        elif age <= 45:
            return "30-45"
        elif age <= 60:
            return "46-60"
        else:
            return "60+"
    
    df['AgeGroup'] = df['Age'].apply(age_group)
    
    # Balance Group
    def balance_group(b):
        if b == 0:
            return "Zero"
        elif b < 100000:
            return "Low"
        else:
            return "High"
    
    df['BalanceGroup'] = df['Balance'].apply(balance_group)
    
    # Tenure Group
    def tenure_group(t):
        if t <= 3:
            return "New"
        elif t <= 7:
            return "Mid"
        else:
            return "Long"
    
    df['TenureGroup'] = df['Tenure'].apply(tenure_group)
    
    return df

df = load_data()
# -------------------------------
# ML MODEL PREPARATION
# -------------------------------

# Select features
features = ['CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender']

X = df[features]
y = df['Exited']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

geo = st.sidebar.multiselect("Select Geography", df['Geography'].unique(), default=df['Geography'].unique())
age = st.sidebar.multiselect("Select Age Group", df['AgeGroup'].unique(), default=df['AgeGroup'].unique())

filtered_df = df[(df['Geography'].isin(geo)) & (df['AgeGroup'].isin(age))]

# -------------------------------
# KPI SECTION
# -------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

churn_rate = filtered_df['Exited'].mean() * 100
avg_balance = filtered_df['Balance'].mean()
total_customers = filtered_df.shape[0]

col1.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col2.metric("Avg Balance", f"{avg_balance:.2f}")
col3.metric("Total Customers", total_customers)

# -------------------------------
# CHARTS SECTION
# -------------------------------
st.subheader("📊 Churn Analysis")

col1, col2 = st.columns(2)

# Geography Chart
with col1:
    st.write("Churn by Geography")
    fig, ax = plt.subplots()
    sns.barplot(x='Geography', y='Exited', data=filtered_df, ax=ax)
    st.pyplot(fig)

# Age Group Chart
with col2:
    st.write("Churn by Age Group")
    fig, ax = plt.subplots()
    sns.barplot(x='AgeGroup', y='Exited', data=filtered_df, ax=ax)
    st.pyplot(fig)

# -------------------------------
# BALANCE ANALYSIS
# -------------------------------
st.subheader("💰 Balance vs Churn")

fig, ax = plt.subplots()
sns.boxplot(x='Exited', y='Balance', data=filtered_df, ax=ax)
st.pyplot(fig)

# -------------------------------
# ACTIVE vs INACTIVE
# -------------------------------
st.subheader("⚡ Engagement Analysis")

fig, ax = plt.subplots()
sns.barplot(x='IsActiveMember', y='Exited', data=filtered_df, ax=ax)
st.pyplot(fig)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))
# -------------------------------
# CHURN PREDICTION
# -------------------------------
st.subheader("🤖 Churn Prediction")

st.write("Enter customer details:")

col1, col2 = st.columns(2)

with col1:
    credit = st.number_input("Credit Score", 300, 900, 600)
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)

with col2:
    products = st.number_input("Number of Products", 1, 4, 1)
    card = st.selectbox("Has Credit Card", [0,1])
    active = st.selectbox("Is Active Member", [0,1])
    salary = st.number_input("Estimated Salary", 10000.0, 200000.0, 50000.0)
    gender = st.selectbox("Gender (Male=1, Female=0)", [1,0])

# Predict button
if st.button("Predict Churn"):
    
    input_data = [[credit, age, tenure, balance,
                   products, card, active, salary, gender]]
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Churn ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Churn ({probability*100:.2f}%)")
