import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AgriExplain AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

DEFAULT_INPUTS = {
    "N": 40.0,
    "P": 50.0,
    "K": 55.0,
    "temperature": 27.0,
    "humidity": 60.0,
    "ph": 6.5,
    "rainfall": 105.0,
}

# =========================================================
# 2. SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "crop_inputs" not in st.session_state:
    st.session_state.crop_inputs = DEFAULT_INPUTS.copy()

# =========================================================
# 3. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model_and_explainer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Make sure models/crop_model.pkl exists."
        )
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}\n"
            "Make sure models/feature_names.pkl exists."
        )

    model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURE_PATH))
    explainer = shap.TreeExplainer(model)
    return model, features, explainer


try:
    model, features, explainer = load_model_and_explainer()
except Exception as exc:
    st.error(f"Could not load the model or SHAP explainer.\n\nDetails: {exc}")
    st.stop()

# Confirm that the saved feature list matches the input form.
missing_defaults = [name for name in features if name not in DEFAULT_INPUTS]
if missing_defaults:
    st.error(
        "The saved model expects feature(s) that are not configured in this app: "
        + ", ".join(missing_defaults)
    )
    st.stop()

# =========================================================
# 4. CUSTOM STYLE
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --forest: #214D38;
        --leaf: #2C7650;
        --lime: #D8E8A5;
        --cream: #F7F5E9;
        --orange: #E98A24;
        --ink: #17231C;
        --muted: #AAB7AE;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(44,118,80,.18), transparent 36%),
            radial-gradient(ellipse at 100% 10%, rgba(233,138,36,.08), transparent 28%),
            #0e1512;
        color: #F4F7F2;
    }

    [data-testid="stHeader"] {
        background: rgba(14, 21, 18, 0.82);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.7rem 1.8rem 1.5rem;
        border: 1px solid rgba(216,232,165,.2);
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(33,77,56,.94), rgba(24,48,36,.9) 60%, rgba(43,66,43,.78));
        box-shadow: 0 18px 45px rgba(0,0,0,.18);
        margin-bottom: 1.5rem;
    }

    .hero:after {
        content: "🌱";
        position: absolute;
        right: 3rem;
        top: -1.7rem;
        font-size: 8rem;
        opacity: .12;
        transform: rotate(-12deg);
        pointer-events: none;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: .8rem;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 54px;
        height: 54px;
        border-radius: 17px;
        background: rgba(216,232,165,.14);
        border: 1px solid rgba(216,232,165,.25);
        font-size: 1.9rem;
    }

    .hero h1 {
        color: #F7F8EF;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
    }

    .hero-subtitle {
        color: #D5E2D5;
        font-size: 1rem;
        margin-top: .85rem;
        max-width: 760px;
        line-height: 1.6;
    }

    .eyebrow {
        display: inline-block;
        color: #E5F1B9;
        background: rgba(216,232,165,.12);
        border: 1px solid rgba(216,232,165,.2);
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: .85rem;
    }

    .section-intro {
        color: #B9C6BC;
        margin-top: -.35rem;
        margin-bottom: 1.2rem;
        line-height: 1.65;
    }

    .info-strip {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin: .5rem 0 1rem;
    }

    .info-chip {
        padding: .45rem .75rem;
        border-radius: 999px;
        border: 1px solid rgba(216,232,165,.18);
        background: rgba(255,255,255,.035);
        color: #D9E4D9;
        font-size: .83rem;
    }

    div[data-testid="stForm"] {
        background: rgba(23, 35, 28, .82);
        border: 1px solid rgba(183, 207, 177, .19);
        border-radius: 22px;
        padding: 1.35rem 1.35rem .75rem;
        box-shadow: 0 14px 35px rgba(0,0,0,.13);
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #E8EFE8;
        font-weight: 650;
    }

    div[data-testid="stNumberInput"] input {
        background: #202D25;
        color: #F6F8F2;
        border-radius: 11px;
    }

    div[data-testid="stNumberInput"] button {
        color: #D8E8A5;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        transition: all .18s ease;
        border: 1px solid rgba(216,232,165,.22);
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        color: #17231C;
        background: linear-gradient(95deg, #D8E8A5, #BBD77B);
        border: none;
        box-shadow: 0 7px 18px rgba(164,196,94,.16);
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(164,196,94,.25);
    }

    .result-card {
        background: linear-gradient(120deg, #173A2A, #1D4A34);
        border: 1px solid rgba(103,219,148,.34);
        padding: 1.5rem 1.6rem;
        border-radius: 20px;
        margin: .6rem 0 1rem;
        box-shadow: 0 14px 32px rgba(0,0,0,.15);
    }

    .result-card .result-label {
        color: #B7D8C2;
        font-size: .82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .1em;
        margin-bottom: .5rem;
    }

    .result-card h2 {
        color: #78E5A5;
        font-size: clamp(2rem, 4vw, 2.8rem);
        margin: 0;
        line-height: 1.15;
    }

    .result-card .result-description {
        color: #D2E4D7;
        margin-top: .65rem;
        font-size: .96rem;
    }

    .confidence-note {
        color: #E7E1B3;
        font-size: .91rem;
        line-height: 1.55;
        padding: .85rem 1rem;
        border: 1px solid rgba(214,184,76,.22);
        border-left: 4px solid #D6B84C;
        background: rgba(214,184,76,.08);
        border-radius: 12px;
        margin-top: .2rem;
    }

    .candidate-card {
        background: linear-gradient(145deg, rgba(35,55,43,.95), rgba(25,38,30,.95));
        border: 1px solid rgba(183,207,177,.17);
        border-radius: 16px;
        padding: 1rem;
        min-height: 112px;
    }

    .candidate-rank {
        color: #D8E8A5;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 750;
    }

    .candidate-name {
        color: #F3F6ED;
        font-size: 1.15rem;
        font-weight: 750;
        margin: .35rem 0;
    }

    .candidate-probability {
        color: #8DE1AA;
        font-size: 1.35rem;
        font-weight: 800;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(183,207,177,.15);
        padding: .85rem 1rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #C0CDC2;
    }

    div[data-testid="stMetricValue"] {
        color: #F4F7F2;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(183,207,177,.17);
        border-radius: 12px;
        background: rgba(255,255,255,.02);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(183,207,177,.14);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer-note {
        color: #91A197;
        font-size: .82rem;
        line-height: 1.6;
        text-align: center;
        padding-top: .7rem;
    }

    @media (max-width: 700px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .hero {
            padding: 1.2rem;
            border-radius: 18px;
        }
        .hero:after {
            right: .5rem;
            font-size: 5rem;
        }
        div[data-testid="stForm"] {
            padding: .8rem .75rem .4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 5. REUSABLE HEADER
# =========================================================
def show_header():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Smart agriculture · Explainable AI</div>
            <div class="brand-row">
                <div class="brand-icon">🌱</div>
                <h1>AgriExplain AI</h1>
            </div>
            <div class="hero-subtitle">
                Turn soil and climate conditions into crop insights you can understand.
                Explore the model's recommendation and the factors behind it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. INPUT PAGE
# =========================================================
if st.session_state.page == "input":
    show_header()
    st.markdown("## Let's understand your field")
    st.markdown(
        '<div class="section-intro">Enter the values from your soil test and local '
        'weather information. We will return a crop prediction, alternative candidates, '
        'and an explanation of the model output.</div>'
        '<div class="info-strip">'
        '<span class="info-chip">🧪 Soil nutrients</span>'
        '<span class="info-chip">🌦️ Climate conditions</span>'
        '<span class="info-chip">🔎 Explainable prediction</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    current = st.session_state.crop_inputs

    with st.form("crop_input_form"):
        left, right = st.columns(2)

        with left:
            N = st.number_input(
                "Nitrogen (N)", min_value=0.0, max_value=200.0,
                value=float(current["N"]), step=1.0
            )
            P = st.number_input(
                "Phosphorus (P)", min_value=0.0, max_value=200.0,
                value=float(current["P"]), step=1.0
            )
            K = st.number_input(
                "Potassium (K)", min_value=0.0, max_value=250.0,
                value=float(current["K"]), step=1.0
            )
            temperature = st.number_input(
                "Temperature (°C)", min_value=-10.0, max_value=60.0,
                value=float(current["temperature"]), step=0.5
            )

        with right:
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0, max_value=100.0,
                value=float(current["humidity"]), step=1.0
            )
            ph = st.number_input(
                "Soil pH", min_value=0.0, max_value=14.0,
                value=float(current["ph"]), step=0.1
            )
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, max_value=500.0,
                value=float(current["rainfall"]), step=1.0
            )

        submitted = st.form_submit_button(
            "🌱 Generate Crop Recommendation", type="primary", use_container_width=True
        )

    st.caption(
        "Research prototype. The input limits are interface safeguards, "
        "not validated agricultural suitability limits."
    )

    if submitted:
        st.session_state.crop_inputs = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
        st.session_state.page = "results"
        st.rerun()


# =========================================================
# 7. RESULTS PAGE
# =========================================================
elif st.session_state.page == "results":
    show_header()

    if st.button("← Edit Agricultural Conditions"):
        st.session_state.page = "input"
        st.rerun()

    input_values = st.session_state.crop_inputs
    input_df = pd.DataFrame(
        [[input_values[name] for name in features]],
        columns=features,
    )

    try:
        predicted_crop = model.predict(input_df)[0]
        probabilities = np.asarray(model.predict_proba(input_df)[0], dtype=float)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    classes = np.asarray(model.classes_)
    if len(classes) != len(probabilities):
        st.error("The model classes do not match its probability output.")
        st.stop()

    ranking_indices = np.argsort(probabilities)[::-1]
    predicted_index = int(np.where(classes == predicted_crop)[0][0])
    confidence = float(probabilities[predicted_index] * 100.0)

    # -----------------------------------------------------
    # PREDICTION SUMMARY
    # -----------------------------------------------------
    st.markdown("## Your field insights")
    st.markdown(
        '<div class="section-intro">Here is the model output for the conditions you entered. '
        'Use the alternatives and explanation to support further investigation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 1 · Leading crop prediction")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Top model candidate</div>
            <h2>🌾 {str(predicted_crop).title()}</h2>
            <div class="result-description">
                Ranked first for the soil and environmental values you provided.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col, note_col = st.columns([1, 2])
    with metric_col:
        st.metric("Model probability", f"{confidence:.2f}%")
    with note_col:
        if confidence < 50:
            st.markdown(
                """
                <div class="confidence-note">
                    <strong>Confidence note</strong> &nbsp; The model is not strongly
                    separating the leading crop from the alternatives for this input.
                    Compare the candidates and validate with local soil, seasonal,
                    and water-availability information.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif confidence < 75:
            st.info(
                "Moderate model confidence. Verify soil, season, water "
                "availability, and local crop suitability."
            )
        else:
            st.success(
                "The model assigns a relatively high probability to this class. "
                "This still does not guarantee crop suitability or yield."
            )

    st.caption(
        "Model probability is an estimate from the trained classifier; it is "
        "not a measured probability of crop success, yield, or profitability."
    )

    # -----------------------------------------------------
    # TOP THREE CANDIDATES
    # -----------------------------------------------------
    st.markdown("### 2 · Compare likely crop candidates")
    st.markdown(
        '<div class="section-intro">These are the model’s highest-ranked classes, '
        'not guaranteed measures of crop success.</div>',
        unsafe_allow_html=True,
    )
    top_indices = ranking_indices[:3]
    top_df = pd.DataFrame({
        "Rank": range(1, len(top_indices) + 1),
        "Crop": [str(classes[i]).title() for i in top_indices],
        "Model probability (%)": [
            round(float(probabilities[i] * 100.0), 2) for i in top_indices
        ],
    })

    candidate_columns = st.columns(len(top_indices))
    for position, (column, (_, row)) in enumerate(zip(candidate_columns, top_df.iterrows())):
        with column:
            st.markdown(
                f"""
                <div class="candidate-card">
                    <div class="candidate-rank">Candidate #{int(row['Rank'])}</div>
                    <div class="candidate-name">{row['Crop']}</div>
                    <div class="candidate-probability">
                        {row['Model probability (%)']:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("View candidate probabilities in a table"):
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------
    with st.expander("Review the input conditions"):
        display_inputs = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(name, name) for name in features],
            "Value": [input_values[name] for name in features],
        })
        st.dataframe(display_inputs, hide_index=True, use_container_width=True)

    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------
    st.markdown("### 3 · Why did the model choose this crop?")
    st.markdown(
        f'<div class="section-intro">SHAP helps explain which inputs influenced the '
        f'model output for <strong>{str(predicted_crop).title()}</strong>. '
        'Green bars push the output toward this class; red bars push it away.</div>',
        unsafe_allow_html=True,
    )

    def extract_class_shap_values(shap_result, class_index, feature_count):
        """Return one SHAP value per feature for the selected class."""
        raw = getattr(shap_result, "values", shap_result)

        # Some SHAP versions return a list containing one array per class.
        if isinstance(raw, list):
            if class_index >= len(raw):
                raise ValueError("SHAP output does not contain the predicted class.")
            arr = np.asarray(raw[class_index])
            if arr.ndim == 2:
                arr = arr[0]
            return np.asarray(arr, dtype=float).reshape(-1)

        arr = np.asarray(raw)

        # Typical multiclass output: samples x features x classes.
        if arr.ndim == 3:
            if arr.shape[1] == feature_count and arr.shape[2] > class_index:
                return np.asarray(arr[0, :, class_index], dtype=float)

            # Alternative output: classes x samples x features.
            if arr.shape[0] > class_index and arr.shape[2] == feature_count:
                return np.asarray(arr[class_index, 0, :], dtype=float)

            # Alternative output: samples x classes x features.
            if arr.shape[0] >= 1 and arr.shape[1] > class_index and arr.shape[2] == feature_count:
                return np.asarray(arr[0, class_index, :], dtype=float)

            raise ValueError(f"Unexpected 3D SHAP output shape: {arr.shape}")

        # Binary/single-output output: samples x features or features.
        if arr.ndim == 2:
            if arr.shape[1] == feature_count:
                return np.asarray(arr[0], dtype=float)
            if arr.shape[0] == feature_count:
                return np.asarray(arr[:, 0], dtype=float)

        if arr.ndim == 1 and arr.shape[0] == feature_count:
            return np.asarray(arr, dtype=float)

        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

    try:
        shap_result = explainer.shap_values(input_df)
        contributions = extract_class_shap_values(
            shap_result, predicted_index, len(features)
        )

        if len(contributions) != len(features):
            raise ValueError(
                f"Got {len(contributions)} SHAP values for {len(features)} features."
            )
        if not np.all(np.isfinite(contributions)):
            raise ValueError("SHAP returned non-finite contribution values.")

        explanation_df = pd.DataFrame({
            "Feature": list(features),
            "SHAP contribution": contributions,
        })
        explanation_df["Absolute contribution"] = (
            explanation_df["SHAP contribution"].abs()
        )
        explanation_df = explanation_df.sort_values(
            "Absolute contribution", ascending=True
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))
        colors = [
            "#2C8C55" if value >= 0 else "#CD5050"
            for value in explanation_df["SHAP contribution"]
        ]
        ax.barh(
            explanation_df["Feature"],
            explanation_df["SHAP contribution"],
            color=colors,
        )
        ax.axvline(0, color="gray", linewidth=0.9)
        ax.set_xlabel("SHAP contribution")
        ax.set_title(f"Feature contributions for {str(predicted_crop).title()}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        display_df = explanation_df[
            ["Feature", "SHAP contribution"]
        ].copy()
        display_df["Feature"] = display_df["Feature"].map(
            lambda name: FEATURE_LABELS.get(name, name)
        )
        display_df["SHAP contribution"] = display_df[
            "SHAP contribution"
        ].round(6)
        display_df = display_df.sort_values(
            "SHAP contribution",
            key=lambda column: column.abs(),
            ascending=False,
        )

        st.subheader("Feature Contribution Details")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption(
            "SHAP describes how the model's output is attributed to features. "
            "It does not prove that a feature causes a crop to grow successfully."
        )

    except Exception as exc:
        st.warning(
            "The crop prediction was generated, but the SHAP explanation "
            f"could not be displayed. Details: {exc}"
        )
        st.caption(
            "Check the installed SHAP version and the output shape for this model."
        )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------
    st.divider()
    st.markdown(
        '<div class="footer-note">AGRIEXPLAIN AI · RESEARCH PROTOTYPE<br>'
        'Validate recommendations with local soil tests, seasonal weather, '
        'water availability, and qualified agricultural advice before making '
        'farming decisions.</div>',
        unsafe_allow_html=True,
    )
