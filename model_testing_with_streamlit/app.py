import pandas as pd
import streamlit as st

# Must be imported before the model is loaded -- this is what registers
# HandleOutlier / AgeTransformer / AgeBinner / ColumnDropper under __main__
# so pickle can resolve them. See Requirement.py for details.
import Requirement as Requirement  # noqa: F401

from pycaret.regression import load_model, predict_model

st.set_page_config(
    page_title="South Korea Daegu Apartment Price Predictor", layout="centered")
st.title("South Korea Daegu")
st.title("Apartment Price Predictor")

# pycaret's load_model appends ".pkl" itself -- give it the path WITHOUT
# the extension. Adjust this path to wherever you place the file relative
# to app.py (e.g. keep the "model/" folder layout from your notebooks).
MODEL_PATH = "model/final_model"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    model = get_model()
except Exception as e:
    st.error(
        "Couldn't load the model. This almost always means the running "
        "Python environment doesn't match what the model was trained with "
        "(scikit-learn 1.2.2, pandas < 2.0, Python 3.10.11). "
        f"\n\nError: {e}"
    )
    st.stop()

st.caption("Fill in the property details and get a predicted sale price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        hallway_type = st.selectbox(
            "Hallway type", ["terraced", "mixed", "corridor"])
        time_to_subway = st.selectbox(
            "Time to subway",
            ["no_bus_stop_nearby", "0-5min", "5min~10min",
                "10min~15min", "15min~20min"],
        )
        subway_station = st.selectbox(
            "Nearest subway station", [
                "no_subway_nearby", "Kyungbuk_uni_hospital", "Chil-sung-market", "Bangoge", "Sin-nam", "Banwoldang", "Myung-duk", "Daegu"
            ],
        )
        year_built = st.number_input(
            "Year built", min_value=1950, max_value=2025, value=2007, step=1)
        size_sqf = st.number_input(
            "Size (sqf)", min_value=100, max_value=10000, value=1334, step=1)

    with col2:
        n_facilities_etc = st.selectbox(
            "Nearby facilities (ETC)", [0, 1, 2, 5])
        n_facilities_public_office = st.selectbox(
            "Nearby facilities (Public office)", [0, 1, 2, 3, 4, 5, 6, 7])
        n_school_university = st.selectbox(
            "Nearby schools (University)", [0, 1, 2, 3, 4, 5])
        n_parkinglot_basement = st.number_input(
            "Parking lot spaces (basement)", min_value=0, max_value=2000, value=605, step=1
        )
        n_facilities_in_apt = st.number_input(
            "Facilities in apartment", min_value=0, max_value=50, value=5, step=1
        )

    submitted = st.form_submit_button("Predict price")

if submitted:
    input_df = pd.DataFrame([{
        "HallwayType": hallway_type,
        "TimeToSubway": time_to_subway,
        "SubwayStation": subway_station,
        "N_FacilitiesNearBy(ETC)": float(n_facilities_etc),
        "N_FacilitiesNearBy(PublicOffice)": float(n_facilities_public_office),
        "N_SchoolNearBy(University)": float(n_school_university),
        "N_Parkinglot(Basement)": float(n_parkinglot_basement),
        "YearBuilt": int(year_built),
        "N_FacilitiesInApt": int(n_facilities_in_apt),
        "Size(sqf)": int(size_sqf),
    }])

    result = predict_model(model, data=input_df)
    predicted_price = result["prediction_label"].iloc[0]

    st.success(
        f"Predicted sale price (South Korea Won): {predicted_price:,.0f}")
    with st.expander("Full model output"):
        st.dataframe(result)
