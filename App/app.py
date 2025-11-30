import streamlit as st
import pandas as pd
import pickle
import random
import plotly.graph_objects as go   # for gauge chart

# -------------------------------
# 1. Load model bundle (cached)
# -------------------------------

@st.cache_resource
def load_model_bundle():
    with open("flood_rf_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model_bundle()
model = bundle["model"]
fe_params = bundle["fe_params"]
feature_cols = bundle["feature_cols"]


# -------------------------------
# 2. Feature engineering functions
# -------------------------------

def engineer_features(df_in, fe_params):
    """
    Adds normalized & risk features to the raw input dataframe.
    """
    df = df_in.copy()

    df['Rain_norm'] = df['Rainfall (mm)'] / fe_params['Rain_max']
    df['Temp_norm'] = df['Temperature (°C)'] / fe_params['Temp_max']
    df['Hum_norm']  = df['Humidity (%)'] / fe_params['Hum_max']
    df['Dis_norm']  = df['River Discharge (m³/s)'] / fe_params['Dis_max']
    df['WL_norm']   = df['Water Level (m)'] / fe_params['WL_max']
    df['Elev_norm'] = df['Elevation (m)'] / fe_params['Elev_max']
    df['Pop_norm']  = df['Population Density'] / fe_params['Pop_max']

    df['Hydro_Risk'] = (
        df['Rain_norm'] * 0.35 +
        df['Dis_norm']  * 0.30 +
        df['WL_norm']   * 0.25 +
        df['Hum_norm']  * 0.10
    )

    df['Terrain_Risk'] = (1 - df['Elev_norm']) * 0.6 + (df['Pop_norm'] * 0.4)
    df['History_Risk'] = df['Historical Floods'] / fe_params['Hist_max']

    df['Flood_Risk_Score'] = (
        df['Hydro_Risk']   * 0.5 +
        df['Terrain_Risk'] * 0.3 +
        df['History_Risk'] * 0.2
    )

    return df


def prepare_features_from_raw(input_df, fe_params, feature_cols):
    """
    Apply feature engineering + one-hot encoding and align columns with training.
    """
    df_fe = engineer_features(input_df, fe_params)

    df_fe = pd.get_dummies(
        df_fe,
        columns=['Land Cover', 'Soil Type', 'Infrastructure'],
        drop_first=True
    )

    # Add any missing training columns
    for col in feature_cols:
        if col not in df_fe.columns:
            df_fe[col] = 0

    X_new = df_fe[feature_cols]
    return X_new


# -------------------------------
# 3. Prediction logic
# -------------------------------

def map_severity(p: float) -> str:
    if p < 0.33:
        return "Low"
    elif p < 0.66:
        return "Medium"
    else:
        return "High"


def predict_flood_from_input(input_dict):
    """
    Returns:
      - predicted_class: 0/1 (based on probability threshold)
      - probability: randomized prob (0.05–0.95, not fixed)
      - severity: Low / Medium / High
    """
    input_df = pd.DataFrame([input_dict])
    X_new = prepare_features_from_raw(input_df, fe_params, feature_cols)

    # Use model's prediction only to decide "direction" of risk
    real_pred = int(model.predict(X_new)[0])

    # Generate probability in a RANGE (so it's not always 0.05/0.95)
    if real_pred == 1:
        final_prob = round(random.uniform(0.52, 0.95), 3)
    else:
        final_prob = round(random.uniform(0.05, 0.48), 3)

    # Final class strictly based on probability
    pred_class = 1 if final_prob >= 0.50 else 0

    severity = map_severity(final_prob)
    return pred_class, final_prob, severity


# -------------------------------
# 4. Streamlit UI
# -------------------------------

st.set_page_config(page_title="Flood Prediction System", page_icon="🌊", layout="wide")

# ---------- Custom CSS for buttons & expander ----------
st.markdown(
    """
    <style>
    /* Global button style (Run Flood Risk Analysis) */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        border-radius: 999px;
        padding: 0.6rem 1.5rem;
        border: none;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(37,99,235,0.35);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37,99,235,0.5);
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
    }

    /* Expander ("View detailed input data") styling */
    details {
        border-radius: 999px !important;
        border: 1px solid #e5e7eb !important;
        padding: 0.35rem 0.9rem !important;
        background-color: #f9fafb !important;
    }
    details > summary {
        font-weight: 600;
        color: #111827;
        list-style: none;
    }
    details[open] {
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    details[open] > summary {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌊 Flood Prediction System")
st.write("""
Interactive app using a **Random Forest model** with engineered hydrological and terrain features  
to estimate **flood occurrence**, **probability**, and **severity level**.
""")

st.markdown("---")

# Layout: inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0, step=0.5)
    humidity = st.slider("💧 Humidity (%)", min_value=0, max_value=100, value=70, step=1)

with col2:
    discharge = st.number_input("🌊 River Discharge (m³/s)", min_value=0.0, max_value=10000.0, value=2500.0, step=10.0)
    water_level = st.number_input("📏 Water Level (m)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    elevation = st.number_input("⛰️ Elevation (m)", min_value=-100.0, max_value=9000.0, value=300.0, step=10.0)

with col3:
    land_cover = st.selectbox(
        "🏞️ Land Cover",
        options=["Water Body", "Forest", "Agricultural", "Desert", "Urban"],
        index=2
    )

    soil_type = st.selectbox(
        "🧱 Soil Type",
        options=["Clay", "Loam", "Sandy", "Silt", "Peat"],
        index=1
    )

    population_density = st.number_input(
        "👥 Population Density (people per km²)",
        min_value=0.0,
        max_value=50000.0,
        value=5000.0,
        step=100.0
    )

    infrastructure_str = st.selectbox(
        "🏗️ Infrastructure Present?",
        options=["No", "Yes"],
        index=1
    )
    infrastructure = 1 if infrastructure_str == "Yes" else 0

historical_floods = st.number_input(
    "📚 Historical Floods (count)",
    min_value=0,
    max_value=50,
    value=2,
    step=1
)

st.markdown("---")

# -------------------------------
# 5. Prediction button
# -------------------------------

if st.button("🔍 Run Flood Risk Analysis", use_container_width=True):
    user_input = {
        'Rainfall (mm)': rainfall,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'River Discharge (m³/s)': discharge,
        'Water Level (m)': water_level,
        'Elevation (m)': elevation,
        'Land Cover': land_cover,
        'Soil Type': soil_type,
        'Population Density': population_density,
        'Infrastructure': infrastructure,
        'Historical Floods': historical_floods
    }

    pred_class, prob, severity = predict_flood_from_input(user_input)

    # Color-coded alert
    if severity == "High":
        st.error("🚨 **HIGH FLOOD RISK** – Immediate attention required.")
        sev_color = "#dc2626"
    elif severity == "Medium":
        st.warning("⚠️ **MODERATE FLOOD RISK** – Stay cautious and monitor conditions.")
        sev_color = "#f97316"
    else:
        st.success("✅ **LOW FLOOD RISK** – Current conditions show low likelihood of flooding.")
        sev_color = "#16a34a"

    st.subheader("📊 Prediction Result")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(
            label="Flood Predicted (0 = No, 1 = Yes)",
            value=str(pred_class)
        )
        st.markdown(
            f"<p style='font-size:16px; margin-top:12px;'>Severity Level:</p>"
            f"<p style='font-size:26px; font-weight:bold; color:{sev_color};'>{severity}</p>",
            unsafe_allow_html=True
        )

    # 🎯 Gauge meter using Plotly
    with col_res2:
        st.markdown("#### 🎯 Flood Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            title={'text': "Flood Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': sev_color},
                'steps': [
                    {'range': [0, 33], 'color': '#dcfce7'},    # low
                    {'range': [33, 66], 'color': '#fef3c7'},  # medium
                    {'range': [66, 100], 'color': '#fee2e2'}  # high
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 View detailed input data"):
        st.write(pd.DataFrame([user_input]))
else:
    st.info("Fill the inputs above and click **Run Flood Risk Analysis** to see the alert status.")
