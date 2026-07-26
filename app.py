import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#003366;
}

h2{
    color:#003366;
}

.sidebar .sidebar-content{
    background:#003366;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:45px;
    background:#003366;
    color:white;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD FILES
# -------------------------------------------------------

@st.cache_resource
def load_pipeline():

    pipeline = joblib.load("churn_pipeline.pkl")

    return (
        pipeline["model"],
        pipeline["preprocessor"],
        pipeline["label_encoder"]
    )

@st.cache_data
def load_dataset():

    return pd.read_csv("cleaned_customer_churn.csv")

@st.cache_data
def load_feature_importance():

    return pd.read_csv("feature_importance.csv")

@st.cache_resource
def load_metrics():

    return joblib.load("model_metrics.pkl")


model, preprocessor, label_encoder = load_pipeline()

df = load_dataset()

feature_importance = load_feature_importance()

metrics = load_metrics()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("📊 Customer Churn")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dashboard",

        "🔮 Prediction",

        "📈 Model Performance",

        "ℹ️ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success(
    "Machine Learning Classification Project"
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if page == "🏠 Home":

    st.title("📊 Customer Churn Prediction System")

    st.markdown("---")

    st.write("""
This application predicts whether a telecom customer is likely to leave the company.

It helps businesses identify customers who are at risk of churning so they can take preventive actions to improve customer retention.
""")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
      "Features",
     len(df.columns)-1
    )

    churn_rate = round(
        (df["Churn"]=="Yes").mean()*100,
        2
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate}%"
    )

    col4.metric(
        "ML Algorithm",
        "Logistic Regression"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Project Workflow")

    st.write("""

✔ Data Collection

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Data Preprocessing

✔ Model Training

✔ Model Evaluation

✔ Deployment using Streamlit
""")

    st.markdown("---")

    st.subheader("Technologies Used")

    st.write("""

• Python

• Pandas

• NumPy

• Scikit-Learn

• Plotly

• Streamlit

• Matplotlib

""")
# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Customer Churn Dashboard")

    st.markdown("---")

    total_customers = len(df)

    churn_customers = len(df[df["Churn"] == "Yes"])

    retained_customers = len(df[df["Churn"] == "No"])

    churn_rate = round((churn_customers / total_customers) * 100, 2)

    avg_monthly = round(df["MonthlyCharges"].mean(), 2)

    avg_tenure = round(df["tenure"].mean(), 2)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Customers",
        total_customers
    )

    c2.metric(
        "Churn Customers",
        churn_customers
    )

    c3.metric(
        "Retention Rate",
        f"{100-churn_rate:.2f}%"
    )

    st.markdown("---")

    c4, c5 = st.columns(2)

    c4.metric(
        "Average Monthly Charges",
        f"£{avg_monthly}"
    )

    c5.metric(
        "Average Tenure",
        f"{avg_tenure} Months"
    )

    st.markdown("---")

    # ------------------------------------------------
    # CHURN DISTRIBUTION
    # ------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        fig = px.pie(
            df,
            names="Churn",
            hole=0.5,
            color="Churn",
            color_discrete_sequence=["green","red"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Contract Type")

        contract = df["Contract"].value_counts().reset_index()

        contract.columns = ["Contract","Count"]

        fig = px.bar(
            contract,
            x="Contract",
            y="Count",
            color="Contract",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ------------------------------------------------
    # INTERNET SERVICE
    # ------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Internet Service")

        internet = df["InternetService"].value_counts().reset_index()

        internet.columns = ["InternetService","Count"]

        fig = px.bar(
            internet,
            x="InternetService",
            y="Count",
            color="InternetService",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        st.subheader("Payment Method")

        payment = df["PaymentMethod"].value_counts().reset_index()

        payment.columns = ["PaymentMethod","Count"]

        fig = px.bar(
            payment,
            x="PaymentMethod",
            y="Count",
            color="PaymentMethod"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ------------------------------------------------
    # MONTHLY CHARGES
    # ------------------------------------------------

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

        st.subheader("Tenure")

        fig = px.histogram(
            df,
            x="tenure",
            nbins=30,
            color_discrete_sequence=["orange"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------

    st.subheader("Top Important Features")

    top = feature_importance.head(15)

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

    st.success("Dashboard Loaded Successfully ✔")
    # -------------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------------

elif page == "🔮 Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown("---")

    st.write(
        "Fill in the customer details below to predict whether the customer is likely to churn."
    )

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT COLUMN
    # -------------------------

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
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # -------------------------
    # RIGHT COLUMN
    # -------------------------

    with col2:

        backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
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

    predict = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )

    # --------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------

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
        # --------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------

    if predict:

        try:

            processed_data = preprocessor.transform(input_df)

            prediction = model.predict(processed_data)

            probability = model.predict_proba(processed_data)

            result = label_encoder.inverse_transform(prediction)[0]

            churn_probability = probability[0][1] * 100

            stay_probability = probability[0][0] * 100

            st.markdown("---")

            st.subheader("Prediction Result")

            if result == "Yes":

                st.error("🚨 Customer is likely to CHURN")

            else:

                st.success("✅ Customer is likely to STAY")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Stay Probability",
                    f"{stay_probability:.2f}%"
                )

            st.markdown("---")

            # ------------------------------------------
            # RISK LEVEL
            # ------------------------------------------

            st.subheader("Risk Level")

            if churn_probability >= 80:

                st.error("🔴 HIGH RISK CUSTOMER")

            elif churn_probability >= 50:

                st.warning("🟠 MEDIUM RISK CUSTOMER")

            else:

                st.success("🟢 LOW RISK CUSTOMER")

            st.markdown("---")

            # ------------------------------------------
            # BUSINESS RECOMMENDATIONS
            # ------------------------------------------

            st.subheader("Recommended Business Actions")

            if churn_probability >= 80:

                st.write("""
✅ Contact the customer immediately.

✅ Offer a special loyalty discount.

✅ Assign a customer support executive.

✅ Recommend a yearly contract.

✅ Provide exclusive benefits.
""")

            elif churn_probability >= 50:

                st.write("""
✅ Offer promotional discounts.

✅ Recommend bundled services.

✅ Encourage longer contracts.

✅ Follow up through customer care.
""")

            else:

                st.write("""
✅ Customer is satisfied.

✅ Continue normal engagement.

✅ Reward with loyalty points.

✅ Maintain service quality.
""")

            st.markdown("---")

            st.subheader("Customer Information")

            st.dataframe(
                input_df,
                use_container_width=True
            )

        except Exception as e:

            st.error("Prediction Failed")

            st.exception(e)
 # -------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------

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
        "ROC AUC",
        f"{metrics['ROC AUC']*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Top Important Features")

    fig = px.bar(
        feature_importance.head(20),
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Model Metrics")

    st.dataframe(pd.DataFrame([metrics]))

# -------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.write("""
### Customer Churn Prediction System

This project predicts whether a telecom customer is likely to churn
using Machine Learning.

The objective is to help businesses identify customers at risk and
take preventive actions to improve customer retention.
""")

    st.markdown("---")

    st.subheader("Project Workflow")

    st.write("""
✔ Data Collection

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Data Preprocessing

✔ Logistic Regression Model

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
            "Streamlit"
        ]
    })

    st.table(tech)

    st.markdown("---")

    st.subheader("Dataset")

    st.info("""
IBM Telco Customer Churn Dataset

• 7043 Customers

• Binary Classification

• Target Variable: Churn
""")

    st.markdown("---")

    st.subheader("Developer")

    st.success("""
Bidyamanjari Jena

B.Tech Computer Science & Engineering

Machine Learning & Data Science
""")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;font-size:15px;color:gray;'>

© 2026 Customer Churn Prediction System

Developed using ❤️ Python | Scikit-Learn | Streamlit

</div>
""",
unsafe_allow_html=True
)           