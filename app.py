import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Ethereum Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)


# Blue Theme

st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #002b55, #005b96);
            color: white;
        }

        /* Main title */
        .main-title {
            font-size: 42px;
            font-weight: 900;
            color: #ffffff;
            text-align: center;
            text-shadow: 2px 2px 8px #00000066;
            margin-top: 10px;
        }

        /* Subtitle */
        .sub-header {
            color: #b3e0ff;
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 25px;
        }

        /* Info box */
        .info-box {
            background-color: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 35px;
            font-size: 18px;
            text-align: center;
            color: #e6f3ff;
            line-height: 1.6;
        }

        /* Feature labels */
        label, .stSlider label, .stSelectbox label {
            color: #e6f3ff !important;
            font-size: 18px !important;
            font-weight: 500 !important;
        }

        /* Result box */
        .result-box {
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# TITLE & SHORT DESCRIPTION

st.markdown("<div class='main-title'>💳 Ethereum Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Detect suspicious Ethereum transactions using AI</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='info-box'>
        Adjust the features below to estimate the probability of fraud in an Ethereum transaction.
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# LOAD MODEL
# ==============================
xgb_package = joblib.load("xgb_package_files.pkl")
model = xgb_package["model"]
encoder = xgb_package["encoder"]
feature_order = xgb_package["features"]
train_min = xgb_package["train_min"]
train_max = xgb_package["train_max"]

categorical_feature = "erc20_most_rec_token_type"
categorical_options = encoder.classes_.tolist()

# ==============================
# INPUT SECTION
# ==============================
st.markdown("### 🧩 Input Transaction Features")
cols = st.columns(2)
input_data = {}

for i, feature in enumerate(feature_order):
    if feature != categorical_feature:
        min_val = train_min[feature]
        max_val = train_max[feature] * 1.2
        default_val = (min_val + max_val) / 2
        with cols[i % 2]:
            input_data[feature] = st.slider(
                f"{feature}",
                float(min_val),
                float(max_val),
                float(default_val)
            )

input_data[categorical_feature] = st.selectbox(
    categorical_feature, options=categorical_options
)

# ==============================
# PREDICTION
# ==============================
input_df = pd.DataFrame([input_data])
input_df[categorical_feature] = encoder.transform(input_df[categorical_feature].astype(str))
input_df = input_df[feature_order]

prob = float(model.predict_proba(input_df)[:, 1][0])
prob_percentage = prob * 100

# Risk levels
if prob < 0.2:
    risk_label = "🟢 Low Risk"
    risk_msg = "This transaction appears safe."
    color = "#00e676"
elif prob < 0.6:
    risk_label = "🟡 Moderate Risk"
    risk_msg = "Some unusual activity detected. Review advised."
    color = "#ffeb3b"
elif prob < 0.85:
    risk_label = "🟠 High Risk"
    risk_msg = "Potential fraud indicators found."
    color = "#ffa726"
else:
    risk_label = "🔴 Very High Risk"
    risk_msg = "This transaction is highly likely to be fraudulent."
    color = "#ff5252"

# ==============================
# DISPLAY RESULTS
# ==============================
st.write("---")
st.markdown("### Prediction Result")

st.progress(min(prob, 1.0))

st.markdown(
    f"""
    <div class='result-box' style='background-color:{color}22; border:1px solid {color}; color:{color};'>
        <p style='font-size:28px; margin-bottom:10px;'>Fraud Probability: <b>{prob_percentage:.2f}%</b></p>
        <p>{risk_label}</p>
        <p>{risk_msg}</p>
    </div>
    """,
    unsafe_allow_html=True
)
