import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1,h2,h3{
    color:#003366;
}

.sidebar .sidebar-content{
    background-color:#003366;
}

.metric-card{
    background-color:white;
    padding:15px;
    border-radius:12px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD FILES
# -----------------------------
pipeline = joblib.load("Models/churn_pipeline.pkl")
model = pipeline["model"]
preprocessor = pipeline["preprocessor"]
label_encoder = pipeline["label_encoder"]

metrics = joblib.load("Models/model_metrics.pkl")

feature_importance = pd.read_csv(
    "Models/feature_importance.csv"
)

df = pd.read_csv("cleaned_customer_churn.csv")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Customer Churn")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔮 Prediction",
        "📈 Model Performance",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Machine Learning Project

Customer Churn Prediction

Algorithm:
Logistic Regression
"""
)

# ======================================================
# HOME PAGE
# ======================================================

if page=="🏠 Home":

    st.title("📊 Customer Churn Prediction System")

    st.markdown("---")

    st.write("""
Customer churn refers to customers leaving a company's services.

This project predicts whether a customer is likely to churn using
Machine Learning.

The application helps businesses identify customers at risk and
take preventive actions to improve customer retention.
""")

    st.image(
        "https://miro.medium.com/max/1400/1*5M0mM6qFj8H8ZLwP0xM6Uw.png",
        use_container_width=True
    )

    st.markdown("## 🎯 Business Objective")

    st.write("""
The objective of this project is to:

- Predict customer churn.
- Reduce customer loss.
- Improve customer retention.
- Help companies make better marketing decisions.
""")

    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    c2.metric(
        "Features",
        df.shape[1]-1
    )

    c3.metric(
        "Algorithm",
        "Logistic Regression"
    )

    churn_rate = round(
        (df["Churn"]=="Yes").mean()*100,
        2
    )

    c4.metric(
        "Churn Rate",
        f"{churn_rate}%"
    )

    st.markdown("---")

    st.subheader("📁 Dataset Information")

    st.write(df.head())

    st.markdown("---")

    st.subheader("⚙ Project Workflow")

    st.write("""
1. Data Collection

2. Data Cleaning

3. Exploratory Data Analysis

4. Feature Engineering

5. Data Preprocessing

6. Model Training

7. Model Evaluation

8. Streamlit Deployment
""")

    st.markdown("---")

    st.subheader("🛠 Tech Stack")

    st.write("""
- Python

- Pandas

- NumPy

- Scikit-Learn

- Streamlit

- Plotly

- Matplotlib

- Seaborn
""")

    st.success("Navigate through the sidebar to explore the application.")
    # ======================================================
# DASHBOARD PAGE
# ======================================================

elif page == "📊 Dashboard":

    st.title("📊 Customer Churn Dashboard")

    st.markdown("---")

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    total_customers = len(df)

    churn_customers = len(df[df["Churn"] == "Yes"])

    retained_customers = len(df[df["Churn"] == "No"])

    churn_rate = round((churn_customers / total_customers) * 100, 2)

    avg_monthly = round(df["MonthlyCharges"].mean(), 2)

    avg_tenure = round(df["tenure"].mean(), 2)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "Customers Churned",
        churn_customers
    )

    c3.metric(
        "Churn Rate",
        f"{churn_rate}%"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Retained Customers",
        retained_customers
    )

    c5.metric(
        "Average Monthly Charges",
        f"${avg_monthly}"
    )

    c6.metric(
        "Average Tenure",
        f"{avg_tenure} Months"
    )

    st.markdown("---")

    # -----------------------------
    # CHURN DISTRIBUTION
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        churn_counts = df["Churn"].value_counts()

        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Contract Distribution")

        fig = px.bar(

            df["Contract"].value_counts().reset_index(),

            x="Contract",

            y="count",

            color="Contract",

            text_auto=True

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------
    # INTERNET SERVICE
    # -----------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Internet Service")

        fig = px.bar(

            df["InternetService"].value_counts().reset_index(),

            x="InternetService",

            y="count",

            color="InternetService",

            text_auto=True

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        st.subheader("Payment Method")

        fig = px.bar(

            df["PaymentMethod"].value_counts().reset_index(),

            x="PaymentMethod",

            y="count",

            color="PaymentMethod"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------
    # MONTHLY CHARGES
    # -----------------------------

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Monthly Charges")

        fig = px.histogram(

            df,

            x="MonthlyCharges",

            nbins=30,

            color_discrete_sequence=["royalblue"]

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col6:

        st.subheader("Tenure Distribution")

        fig = px.histogram(

            df,

            x="tenure",

            nbins=30,

            color_discrete_sequence=["green"]

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------
    # CHURN BY CONTRACT
    # -----------------------------
    st.subheader("Churn by Contract Type")

    contract = (

        df.groupby(
            ["Contract", "Churn"]
        )

        .size()

        .reset_index(name="Count")

    )

    fig = px.bar(

        contract,

        x="Contract",

        y="Count",

        color="Churn",

        barmode="group",

        text_auto=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # CHURN BY INTERNET SERVICE
    # -----------------------------

    st.subheader("Churn by Internet Service")

    internet = (

        df.groupby(
            ["InternetService", "Churn"]
        )

        .size()

        .reset_index(name="Count")

    )

    fig = px.bar(

        internet,

        x="InternetService",

        y="Count",

        color="Churn",

        barmode="group",

        text_auto=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------

    st.subheader("Top 15 Important Features")

    top_features = feature_importance.head(15)

    fig = px.bar(

        top_features,

        x="Coefficient",

        y="Feature",

        orientation="h",

        color="Coefficient",

        color_continuous_scale="Blues"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.success(
        "Dashboard Loaded Successfully ✅"
    )
    # ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "🔮 Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown("---")

    st.write("Enter customer details to predict whether the customer is likely to churn.")

    col1, col2 = st.columns(2)

    # ----------------------------
    # LEFT COLUMN
    # ----------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    # ----------------------------
    # RIGHT COLUMN
    # ----------------------------

    with col2:

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=850.0
        )

    st.markdown("---")

    # ----------------------------
    # PREDICTION BUTTON
    # ----------------------------

    if st.button("Predict Churn"):

        input_df = pd.DataFrame({

            "gender":[gender],

            "SeniorCitizen":[senior],

            "Partner":[partner],

            "Dependents":[dependents],

            "tenure":[tenure],

            "PhoneService":[phone],

            "MultipleLines":[multiple],

            "InternetService":[internet],

            "OnlineSecurity":[security],

            "OnlineBackup":[backup],

            "DeviceProtection":[protection],

            "TechSupport":[support],

            "StreamingTV":[tv],

            "StreamingMovies":[movies],

            "Contract":[contract],

            "PaperlessBilling":[paperless],

            "PaymentMethod":[payment],

            "MonthlyCharges":[monthly],

            "TotalCharges":[total]

        })

        processed = preprocessor.transform(input_df)

        prediction = model.predict(processed)

        probability = model.predict_proba(processed)

        result = label_encoder.inverse_transform(prediction)[0]

        churn_probability = probability[0][1] * 100

        stay_probability = probability[0][0] * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        if result == "Yes":

            st.error("Customer is likely to CHURN")

        else:

            st.success("Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )

        st.markdown("---")

        # ----------------------------
        # RISK LEVEL
        # ----------------------------

        if churn_probability >= 80:

            st.error("🔴 High Risk Customer")

        elif churn_probability >= 50:

            st.warning("🟠 Medium Risk Customer")

        else:

            st.success("🟢 Low Risk Customer")

        st.markdown("---")

        # ----------------------------
        # BUSINESS RECOMMENDATION
        # ----------------------------

        st.subheader("Business Recommendation")

        if churn_probability >= 80:

            st.write("""
✅ Offer a special discount.

✅ Contact customer personally.

✅ Provide premium customer support.

✅ Offer loyalty rewards.

✅ Recommend long-term contract.
""")

        elif churn_probability >= 50:

            st.write("""
✅ Send promotional offers.

✅ Recommend bundled services.

✅ Encourage yearly subscription.

✅ Follow up with customer support.
""")

        else:

            st.write("""
✅ Customer is likely to stay.

✅ Continue regular engagement.

✅ Offer reward points.

✅ Maintain current service quality.
""")

        st.markdown("---")

        st.subheader("Customer Summary")

        st.dataframe(input_df)
        # ======================================================
# MODEL PERFORMANCE
# ======================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy']*100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{metrics['Precision']*100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{metrics['Recall']*100:.2f}%"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "F1 Score",
        f"{metrics['F1 Score']*100:.2f}%"
    )

    c5.metric(
        "ROC-AUC",
        f"{metrics['ROC AUC']*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Feature Importance")

    top = feature_importance.head(20)

    fig = px.bar(
        top,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Summary")

    st.success(
        """
        ✔ Model : Logistic Regression

        ✔ Classification Problem

        ✔ Target : Customer Churn

        ✔ Accuracy : 80%

        ✔ Production Ready
        """
    )

    st.markdown("---")

    st.write(metrics)

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("---")

    st.header("Customer Churn Prediction System")

    st.write("""
Customer churn prediction is a Machine Learning classification project
that predicts whether a customer is likely to discontinue a service.

Businesses use churn prediction to identify at-risk customers and
implement targeted retention strategies.
""")

    st.markdown("---")

    st.subheader("Project Objectives")

    st.write("""
- Predict customer churn.

- Improve customer retention.

- Reduce revenue loss.

- Support business decision making.

- Provide actionable recommendations.
""")

    st.markdown("---")

    st.subheader("Machine Learning Workflow")

    st.write("""
✔ Data Collection

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Data Preprocessing

✔ Feature Engineering

✔ Model Building

✔ Model Evaluation

✔ Streamlit Deployment
""")

    st.markdown("---")

    st.subheader("Technologies Used")

    tech = pd.DataFrame({

        "Technology":[

            "Python",

            "Pandas",

            "NumPy",

            "Scikit-Learn",

            "Plotly",

            "Matplotlib",

            "Seaborn",

            "Streamlit"

        ]

    })

    st.table(tech)

    st.markdown("---")

    st.subheader("Dataset")

    st.info(
        """
        IBM Telco Customer Churn Dataset

        Records : 7,032

        Features : 20

        Target : Churn
        """
    )

    st.markdown("---")

    st.subheader("Model Used")

    st.success(
        """
        Logistic Regression

        Accuracy : 80.38%

        Precision : 64.85%

        Recall : 57.22%

        F1 Score : 60.80%

        ROC-AUC : 83.59%
        """
    )

    st.markdown("---")

    st.subheader("Project Highlights")

    st.write("""
✔ Interactive Dashboard

✔ Customer Churn Prediction

✔ Probability Estimation

✔ Business Recommendations

✔ Model Performance Visualization

✔ Feature Importance Analysis

✔ Professional Streamlit Interface
""")

    st.markdown("---")

    st.subheader("Developed By")

    st.write("""
**Bidyamanjari Jena**

B.Tech Computer Science & Engineering

Machine Learning & Data Science Enthusiast
""")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;font-size:15px;'>

© 2026 Customer Churn Prediction System

Developed using ❤️ with Python, Scikit-Learn & Streamlit

</div>
""",
unsafe_allow_html=True
)