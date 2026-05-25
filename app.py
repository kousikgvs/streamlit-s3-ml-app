import os
import numpy as np
import streamlit as st
import joblib
from s3_operations import download_file

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
MODEL_PATH   = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")


def models_present() -> bool:
    return os.path.isfile(MODEL_PATH) and os.path.isfile(SCALER_PATH)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def download_models_from_s3() -> bool:
    """Download both artifacts from S3. Returns True on full success."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    ok_model = download_file("models", "random_forest_model.pkl")
    ok_scaler = download_file("models", "scaler.pkl")
    return ok_model and ok_scaler

# ── Encoding maps (matches LabelEncoder alphabetical order from training) ─────
GENDER_MAP        = {"Male": 1, "Female": 0}
MARRIED_MAP       = {"Yes": 1, "No": 0}
DEPENDENTS_MAP    = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP     = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
PROPERTY_MAP      = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.markdown("Fill in the applicant details below and click **Predict**.")

# ── Model download gate ───────────────────────────────────────────────────────
# Predict button stays disabled until models are downloaded locally from S3.
with st.sidebar:
    st.header("Model artifacts")
    if models_present():
        st.success("✅ Models are available locally.")
        if st.button("🔄 Re-download from S3"):
            load_models.clear()
            with st.spinner("Downloading models from S3..."):
                ok = download_models_from_s3()
            if ok:
                st.toast("Models re-downloaded ✅", icon="✅")
                st.rerun()
            else:
                st.error("❌ Failed to download one or more model files from S3.")
    else:
        st.warning("⚠️ Models not found locally. Download them before predicting.")
        if st.button("⬇️ Download models from S3", type="primary"):
            with st.spinner("Downloading models from S3..."):
                ok = download_models_from_s3()
            if ok:
                st.toast("Models downloaded ✅", icon="✅")
                st.rerun()
            else:
                st.error("❌ Failed to download one or more model files from S3.")

models_ready = models_present()

if not models_ready:
    st.info("ℹ️ Open the sidebar and click **Download models from S3** to enable prediction.")
    model, scaler = None, None
else:
    model, scaler = load_models()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender",          ["Male", "Female"],              index=0)
        married         = st.selectbox("Married",         ["Yes", "No"],                   index=0)
        dependents      = st.selectbox("Dependents",      ["0", "1", "2", "3+"],           index=0)
        education       = st.selectbox("Education",       ["Graduate", "Not Graduate"],     index=0)
        self_employed   = st.selectbox("Self Employed",   ["No", "Yes"],                   index=0)
        property_area   = st.selectbox("Property Area",   ["Urban", "Semiurban", "Rural"], index=0)

    with col2:
        applicant_income    = st.number_input("Applicant Income ($)",    min_value=0, value=5000,  step=100)
        coapplicant_income  = st.number_input("Coapplicant Income ($)",  min_value=0, value=1500,  step=100)
        loan_amount         = st.number_input("Loan Amount (thousands)", min_value=1, value=150,   step=10)
        loan_amount_term    = st.number_input("Loan Amount Term (days)", min_value=1, value=360,   step=12)
        credit_history      = st.selectbox("Credit History",
                                           options=[1.0, 0.0],
                                           format_func=lambda x: "Has Credit History (1)" if x == 1.0 else "No Credit History (0)",
                                           index=0)

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True,
        disabled=not models_ready,
        help=None if models_ready else "Download the models from the sidebar first.",
    )

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Encode inputs
    features = np.array([[
        GENDER_MAP[gender],
        MARRIED_MAP[married],
        DEPENDENTS_MAP[dependents],
        EDUCATION_MAP[education],
        SELF_EMPLOYED_MAP[self_employed],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        PROPERTY_MAP[property_area],
    ]])

    # 4. Scale and predict
    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]

    # 5. Display result
    st.divider()
    if prediction == 1:
        st.success(f"✅ **Loan Approved!**  (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"❌ **Loan Rejected.**  (Confidence: {probability[0]*100:.1f}%)")

    with st.expander("Prediction details"):
        st.write(f"- Approval probability : **{probability[1]*100:.2f}%**")
        st.write(f"- Rejection probability: **{probability[0]*100:.2f}%**")
        st.write(f"- Raw prediction value : `{prediction}`")
