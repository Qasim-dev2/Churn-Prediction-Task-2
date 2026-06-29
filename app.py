from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
RESULTS_PATH = BASE_DIR / "results" / "model_comparison.csv"
GRAPHS_DIR = BASE_DIR / "graphs"
EVALUATION_DIR = GRAPHS_DIR / "evaluation"

FEATURE_ORDER = [
    "Age",
    "Gender",
    "City",
    "Subscription_Type",
    "Monthly_Spending",
    "Tenure",
    "Number_of_Purchases",
    "Customer_Support_Requests",
    "Login_Frequency",
    "Satisfaction_Score",
    "Payment_Delay",
    "Contract_Length",
    "Total_Spending",
    "Days_Since_Last_Activity",
]

GENDER_ENCODING = {"Female": 0, "Male": 1}
CITY_ENCODING = {
    "Faisalabad": 0,
    "Islamabad": 1,
    "Karachi": 2,
    "Lahore": 3,
    "Multan": 4,
    "Peshawar": 5,
    "Quetta": 6,
    "Rawalpindi": 7,
}
SUBSCRIPTION_ENCODING = {"Basic": 0, "Premium": 1, "Standard": 2}
LOGIN_ENCODING = {"High": 0, "Low": 1, "Medium": 2}
CONTRACT_ENCODING = {"Monthly": 0, "Quarterly": 1, "Yearly": 2}

EDA_IMAGES = [
    ("Churn Distribution", GRAPHS_DIR / "churn_distribution.png"),
    ("Age Distribution", GRAPHS_DIR / "age_distribution.png"),
    ("Gender vs Churn", GRAPHS_DIR / "gender_vs_churn.png"),
    ("Subscription vs Churn", GRAPHS_DIR / "subscription_vs_churn.png"),
    ("Monthly Spending vs Churn", GRAPHS_DIR / "monthly_spending_vs_churn.png"),
    ("Satisfaction vs Churn", GRAPHS_DIR / "satisfaction_vs_churn.png"),
    ("Login Frequency vs Churn", GRAPHS_DIR / "login_frequency_vs_churn.png"),
    ("Support Requests vs Churn", GRAPHS_DIR / "support_requests_vs_churn.png"),
    ("Correlation Heatmap", GRAPHS_DIR / "correlation_heatmap.png"),
]

EVALUATION_IMAGES = [
    ("Confusion Matrix", EVALUATION_DIR / "confusion_matrix.png"),
    ("Model Comparison F1 Score", EVALUATION_DIR / "model_comparison_f1.png"),
    ("Feature Importance", EVALUATION_DIR / "feature_importance.png"),
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def build_model_input(values):
    encoded = {
        "Age": values["Age"],
        "Gender": GENDER_ENCODING[values["Gender"]],
        "City": CITY_ENCODING[values["City"]],
        "Subscription_Type": SUBSCRIPTION_ENCODING[values["Subscription_Type"]],
        "Monthly_Spending": values["Monthly_Spending"],
        "Tenure": values["Tenure"],
        "Number_of_Purchases": values["Number_of_Purchases"],
        "Customer_Support_Requests": values["Customer_Support_Requests"],
        "Login_Frequency": LOGIN_ENCODING[values["Login_Frequency"]],
        "Satisfaction_Score": values["Satisfaction_Score"],
        "Payment_Delay": values["Payment_Delay"],
        "Contract_Length": CONTRACT_ENCODING[values["Contract_Length"]],
        "Total_Spending": values["Total_Spending"],
        "Days_Since_Last_Activity": values["Days_Since_Last_Activity"],
    }
    return pd.DataFrame([encoded], columns=FEATURE_ORDER)


def get_risk_factors(values):
    risk_factors = []

    if values["Satisfaction_Score"] <= 2:
        risk_factors.append("Low satisfaction score")
    if values["Customer_Support_Requests"] >= 6:
        risk_factors.append("High number of customer support requests")
    if values["Login_Frequency"] == "Low":
        risk_factors.append("Low login frequency")
    if values["Payment_Delay"] > 15:
        risk_factors.append("High payment delay")
    if values["Tenure"] <= 6:
        risk_factors.append("Low customer tenure")
    if values["Days_Since_Last_Activity"] > 45:
        risk_factors.append("Customer inactive for a long time")
    if values["Contract_Length"] == "Monthly":
        risk_factors.append("Monthly contract customers may churn more easily")

    return risk_factors


def metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="liquid-card metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{detail}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(kicker, title, text):
    st.markdown(
        f"""
        <div class="section-heading">
            <span>{kicker}</span>
            <h2>{title}</h2>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_image_gallery(images, columns_count=2):
    for index in range(0, len(images), columns_count):
        columns = st.columns(columns_count)
        for column, (caption, image_path) in zip(columns, images[index : index + columns_count]):
            with column:
                if image_path.exists():
                    st.image(str(image_path), caption=caption, width="stretch")
                else:
                    st.warning(f"{caption} image not found. Run the analysis scripts first.")


st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #132034;
        --muted: #516179;
        --primary: #1f8fff;
        --primary-soft: rgba(82, 169, 255, 0.22);
        --aqua: #23c9d7;
        --pink: #f05aa6;
        --violet: #7257ff;
        --glass: rgba(255, 255, 255, 0.72);
        --glass-strong: rgba(255, 255, 255, 0.88);
        --line: rgba(19, 32, 52, 0.14);
        --shadow: 0 24px 70px rgba(29, 89, 132, 0.16);
    }

    .stApp {
        background:
            linear-gradient(120deg, rgba(82, 169, 255, 0.20), rgba(238, 248, 255, 0) 34%),
            linear-gradient(240deg, rgba(255, 146, 207, 0.18), rgba(238, 248, 255, 0) 38%),
            linear-gradient(160deg, #eef8ff 0%, #e7fbff 42%, #fff4fb 76%, #f4f7ff 100%);
        color: var(--ink);
        font-family: "Avenir Next", "Segoe UI", sans-serif;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.18) 0 1px, transparent 1px 18px),
            linear-gradient(180deg, rgba(255, 255, 255, 0.68), rgba(255, 255, 255, 0.08));
        mix-blend-mode: soft-light;
        opacity: 0.9;
    }

    [data-testid="stHeader"] {
        background: rgba(238, 248, 255, 0.68);
        border-bottom: 1px solid rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(18px);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(229, 250, 255, 0.76)),
            linear-gradient(135deg, rgba(82, 169, 255, 0.22), rgba(255, 146, 207, 0.12));
        border-right: 1px solid rgba(255, 255, 255, 0.7);
        box-shadow: 20px 0 60px rgba(31, 143, 255, 0.10);
        backdrop-filter: blur(22px);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ink) !important;
        font-weight: 800;
    }

    .sidebar-title {
        padding: 1rem 1rem 0.95rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.78);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.54)),
            linear-gradient(135deg, rgba(82, 169, 255, 0.18), rgba(255, 146, 207, 0.12));
        box-shadow: 0 16px 42px rgba(31, 143, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
    }

    .sidebar-title span,
    .sidebar-section-title {
        color: #2875bb !important;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .sidebar-title h2 {
        margin: 0.25rem 0 0.35rem;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.8rem;
        line-height: 1;
    }

    .sidebar-title p {
        margin: 0;
        color: #3c4f68 !important;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .sidebar-section-title {
        display: block;
        margin: 1.1rem 0 0.45rem;
    }

    .sidebar-summary {
        margin: 1rem 0 1.15rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.78);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 14px 34px rgba(31, 143, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
    }

    .sidebar-summary span {
        display: block;
        color: #516179 !important;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-summary strong {
        display: block;
        margin-top: 0.28rem;
        color: var(--ink) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.85rem;
        line-height: 1;
    }

    .sidebar-summary small {
        display: block;
        margin-top: 0.3rem;
        color: #496078 !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #304057 !important;
        font-weight: 650;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid rgba(19, 32, 52, 0.18) !important;
        border-radius: 18px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.92), 0 10px 30px rgba(82, 169, 255, 0.12);
    }

    [data-testid="stSidebar"] [role="slider"] {
        background: linear-gradient(135deg, var(--primary), var(--pink)) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 8px 20px rgba(31, 143, 255, 0.25);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2.7rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        min-height: 260px;
        padding: clamp(1.6rem, 4vw, 3rem);
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.74);
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.42)),
            linear-gradient(120deg, rgba(82, 169, 255, 0.22), rgba(94, 228, 228, 0.16), rgba(255, 146, 207, 0.18));
        box-shadow: var(--shadow), inset 0 1px 1px rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(24px) saturate(150%);
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: -45% -15% auto 42%;
        height: 160%;
        transform: rotate(18deg);
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.52), transparent);
        animation: liquidShimmer 8s ease-in-out infinite;
    }

    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 820px;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(255, 255, 255, 0.8);
        color: #265b88;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.98);
    }

    .hero-shell h1 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.3rem, 5vw, 4.7rem);
        line-height: 0.98;
        color: var(--ink);
    }

    .hero-shell p {
        max-width: 720px;
        margin: 1rem 0 0;
        color: #34445c;
        font-size: 1.05rem;
        line-height: 1.65;
        font-weight: 550;
    }

    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.4rem;
    }

    .hero-stats span {
        padding: 0.58rem 0.82rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(255, 255, 255, 0.78);
        color: #1d3e60;
        font-weight: 800;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.98);
    }

    .liquid-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 26px;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.84), rgba(255, 255, 255, 0.52)),
            linear-gradient(135deg, rgba(82, 169, 255, 0.16), rgba(255, 146, 207, 0.10));
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(22px) saturate(150%);
    }

    .liquid-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.55), transparent 28%, transparent 72%, rgba(255, 255, 255, 0.35));
        pointer-events: none;
    }

    .metric-card {
        min-height: 136px;
        padding: 1.15rem 1.25rem;
    }

    .metric-card span {
        position: relative;
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .metric-card strong {
        position: relative;
        display: block;
        margin-top: 0.32rem;
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1;
    }

    .metric-card small {
        position: relative;
        display: block;
        margin-top: 0.55rem;
        color: #496078;
        font-weight: 650;
    }

    .section-heading {
        margin: 1.9rem 0 0.9rem;
    }

    .section-heading span {
        color: #2875bb;
        font-size: 0.76rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .section-heading h2 {
        margin: 0.15rem 0 0.25rem;
        font-size: clamp(1.8rem, 3vw, 2.75rem);
        line-height: 1.1;
    }

    .section-heading p {
        margin: 0;
        max-width: 780px;
        color: #3c4f68;
        font-size: 1rem;
        line-height: 1.62;
        font-weight: 540;
    }

    .prediction-panel {
        padding: 1.4rem;
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.72);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(22px) saturate(150%);
    }

    .prediction-panel h3 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(1.7rem, 3vw, 2.55rem);
        line-height: 1.05;
    }

    .prediction-panel p {
        margin: 0.75rem 0 0;
        color: #33465f;
        line-height: 1.55;
        font-weight: 580;
    }

    .prediction-panel.safe {
        background: linear-gradient(135deg, rgba(221, 255, 247, 0.88), rgba(255, 255, 255, 0.66));
    }

    .prediction-panel.danger {
        background: linear-gradient(135deg, rgba(255, 235, 246, 0.9), rgba(255, 255, 255, 0.66));
    }

    .probability-badge {
        display: inline-flex;
        margin-top: 1rem;
        padding: 0.6rem 0.85rem;
        border-radius: 999px;
        color: #ffffff;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary), var(--pink));
        box-shadow: 0 14px 28px rgba(240, 90, 166, 0.22);
    }

    .risk-box {
        padding: 1.2rem;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(255, 255, 255, 0.78);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(20px);
    }

    .risk-box h3 {
        margin-top: 0;
        margin-bottom: 0.8rem;
    }

    .risk-pill {
        display: inline-flex;
        align-items: center;
        margin: 0.28rem;
        padding: 0.62rem 0.82rem;
        border-radius: 999px;
        background: rgba(238, 248, 255, 0.9);
        border: 1px solid rgba(82, 169, 255, 0.28);
        color: #17314d;
        font-weight: 760;
    }

    .stButton > button {
        border: 0 !important;
        border-radius: 999px !important;
        padding: 0.78rem 1.35rem !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, var(--primary), var(--aqua) 48%, var(--pink)) !important;
        box-shadow: 0 16px 34px rgba(31, 143, 255, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.42) !important;
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        filter: saturate(1.08);
        box-shadow: 0 20px 42px rgba(240, 90, 166, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 24px;
        box-shadow: var(--shadow);
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.76) !important;
        border-radius: 22px !important;
        background: rgba(255, 255, 255, 0.62) !important;
        box-shadow: var(--shadow);
    }

    [data-testid="stImage"] img {
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 22px 58px rgba(31, 143, 255, 0.14);
    }

    [data-testid="stImageCaption"] {
        color: var(--ink) !important;
        font-weight: 800;
        text-align: left;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.58);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.96);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        color: #35506b;
        font-weight: 850;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, var(--primary), var(--aqua));
    }

    @keyframes liquidShimmer {
        0%, 100% {
            transform: translateX(-24%) rotate(18deg);
            opacity: 0.2;
        }
        50% {
            transform: translateX(16%) rotate(18deg);
            opacity: 0.55;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .hero-shell::after,
        .stButton > button {
            animation: none;
            transition: none;
        }
    }

    @media (max-width: 760px) {
        .hero-shell {
            padding: 1.35rem;
            border-radius: 24px;
        }

        .metric-card {
            min-height: 118px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-content">
            <div class="eyebrow">ML-2 Internship Project</div>
            <h1>Customer Churn Prediction System</h1>
            <p>
                A polished churn-risk dashboard with model-backed prediction, clear customer signals,
                EDA visuals, and evaluated Random Forest results.
            </p>
            <div class="hero-stats">
                <span>2,000 clean records</span>
                <span>Random Forest model</span>
                <span>F1 Score 0.9181</span>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files were not found. Run `python train_model.py` from the main project folder first.")
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            <span>Customer Inputs</span>
            <h2>Prediction Panel</h2>
            <p>Adjust the customer values below. The app uses these exact values for churn prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<span class='sidebar-section-title'>Profile</span>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    city = st.selectbox(
        "City",
        ["Faisalabad", "Islamabad", "Karachi", "Lahore", "Multan", "Peshawar", "Quetta", "Rawalpindi"],
    )

    st.markdown("<span class='sidebar-section-title'>Subscription</span>", unsafe_allow_html=True)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    monthly_spending = st.number_input("Monthly Spending", min_value=500, max_value=6000, value=2000, step=100)
    tenure = st.slider("Tenure", 1, 60, 12)
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])

    total_spending = monthly_spending * tenure
    st.markdown(
        f"""
        <div class="sidebar-summary">
            <span>Total Spending</span>
            <strong>{total_spending:,.0f}</strong>
            <small>Monthly Spending x Tenure</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<span class='sidebar-section-title'>Behavior</span>", unsafe_allow_html=True)
    number_of_purchases = st.slider("Number of Purchases", 1, 80, 10)
    login_frequency = st.selectbox("Login Frequency", ["Low", "Medium", "High"])
    days_since_last_activity = st.slider("Days Since Last Activity", 1, 90, 10)

    st.markdown("<span class='sidebar-section-title'>Risk Signals</span>", unsafe_allow_html=True)
    support_requests = st.slider("Customer Support Requests", 0, 10, 2)
    satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
    payment_delay = st.slider("Payment Delay", 0, 30, 5)

customer_values = {
    "Age": age,
    "Gender": gender,
    "City": city,
    "Subscription_Type": subscription_type,
    "Monthly_Spending": monthly_spending,
    "Tenure": tenure,
    "Number_of_Purchases": number_of_purchases,
    "Customer_Support_Requests": support_requests,
    "Login_Frequency": login_frequency,
    "Satisfaction_Score": satisfaction_score,
    "Payment_Delay": payment_delay,
    "Contract_Length": contract_length,
    "Total_Spending": total_spending,
    "Days_Since_Last_Activity": days_since_last_activity,
}

input_data = build_model_input(customer_values)
summary_data = pd.DataFrame([customer_values])

predict_tab, eda_tab, evaluation_tab = st.tabs(["Predict", "EDA Graphs", "Evaluation Results"])

with predict_tab:
    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        metric_card("Monthly Spending", f"{monthly_spending:,.0f}", "Current customer spend")
    with top_mid:
        metric_card("Total Spending", f"{total_spending:,.0f}", "Monthly spending x tenure")
    with top_right:
        metric_card("Tenure", f"{tenure} months", "Customer relationship age")

    section_intro(
        "Customer Profile",
        "Input Summary",
        "The model receives the same ordered features used during training, with total spending calculated automatically.",
    )
    st.dataframe(summary_data, width="stretch", hide_index=True)

    with st.expander("Encoded model input"):
        st.dataframe(input_data, width="stretch", hide_index=True)

    if st.button("Predict Churn", type="primary"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        churn_probability = model.predict_proba(input_scaled)[0][1]

        status_class = "danger" if prediction == 1 else "safe"
        prediction_text = "Likely to churn" if prediction == 1 else "Not likely to churn"
        confidence_text = "Retention attention recommended" if prediction == 1 else "Customer profile looks stable"

        st.markdown(
            f"""
            <div class="prediction-panel {status_class}">
                <h3>{prediction_text}</h3>
                <p>{confidence_text}</p>
                <div class="probability-badge">Churn Probability: {churn_probability * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        risk_factors = get_risk_factors(customer_values)
        if risk_factors:
            factors_html = "".join(f"<span class='risk-pill'>{factor}</span>" for factor in risk_factors)
        else:
            factors_html = "<span class='risk-pill'>No major churn risk factors found</span>"

        st.markdown(
            f"""
            <div class="risk-box">
                <h3>Important Contributing Factors</h3>
                {factors_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

with eda_tab:
    section_intro(
        "Exploratory Analysis",
        "Customer Behavior Graphs",
        "Generated EDA visuals for churn distribution, demographics, spending, satisfaction, login behavior, support requests, and feature correlation.",
    )
    show_image_gallery(EDA_IMAGES)

with evaluation_tab:
    section_intro(
        "Model Evaluation",
        "Evaluated Results",
        "Final model artifacts and evaluation visuals created from the saved model, scaler, and test split.",
    )

    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH)
        st.dataframe(results_df.round(4), width="stretch", hide_index=True)
    else:
        st.warning("Model comparison file not found. Run `python train_model.py` first.")

    show_image_gallery(EVALUATION_IMAGES)
