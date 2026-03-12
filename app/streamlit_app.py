import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------
# DARK MODE STYLE
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }

    div.stButton > button:first-child:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOAD MODEL
# -------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# HEADER
# -------------------------
st.title("🏠 AI House Price Prediction Dashboard")
st.write("Interactive ML dashboard using Linear Regression")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("House Features")

# longitude = st.sidebar.number_input("Longitude")
# latitude = st.sidebar.number_input("Latitude")
# housing_age = st.sidebar.number_input("Housing Median Age")
# total_rooms = st.sidebar.number_input("Total Rooms")
# total_bedrooms = st.sidebar.number_input("Total Bedrooms")
# population = st.sidebar.number_input("Population")
# households = st.sidebar.number_input("Households")
# median_income = st.sidebar.number_input("Median Income")


longitude = st.sidebar.number_input(
    "Longitude",
    value=None,
    placeholder="Enter longitude"
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=None,
    placeholder="Enter latitude"
)

housing_age = st.sidebar.number_input(
    "Housing Median Age",
    value=None,
    placeholder="Enter housing age"
)

total_rooms = st.sidebar.number_input(
    "Total Rooms",
    value=None,
    placeholder="Enter total rooms"
)

total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms",
    value=None,
    placeholder="Enter bedrooms"
)

population = st.sidebar.number_input(
    "Population",
    value=None,
    placeholder="Enter population"
)

households = st.sidebar.number_input(
    "Households",
    value=None,
    placeholder="Enter households"
)

median_income = st.sidebar.number_input(
    "Median Income",
    value=None,
    placeholder="Enter income"
)

ocean = st.sidebar.selectbox(
    "Ocean Proximity",
    ["INLAND","NEAR BAY","NEAR OCEAN","ISLAND","<1H OCEAN"]
)

# -------------------------
# ONE HOT ENCODING
# -------------------------
INLAND = NEAR_BAY = NEAR_OCEAN = ISLAND = LESS_1H_OCEAN = 0

if ocean == "INLAND":
    INLAND = 1
elif ocean == "NEAR BAY":
    NEAR_BAY = 1
elif ocean == "NEAR OCEAN":
    NEAR_OCEAN = 1
elif ocean == "ISLAND":
    ISLAND = 1
else:
    LESS_1H_OCEAN = 1

# -------------------------
# MAIN LAYOUT
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# PREDICTION SECTION
# -------------------------
with col1:

    # st.subheader("Prediction Result")

    # if st.button("Predict House Price"):

    #     features = np.array([[longitude, latitude, housing_age,
    #                           total_rooms, total_bedrooms,
    #                           population, households, median_income,
    #                           INLAND, NEAR_BAY, NEAR_OCEAN, ISLAND, LESS_1H_OCEAN]])

    #     prediction = model.predict(features)[0]

    #     st.success(f"Predicted Price: ${prediction:,.2f}")

    if st.button("Predict House Price"):

    # validation step
        if None in [longitude, latitude, housing_age, total_rooms,
                    total_bedrooms, population, households, median_income]:

            st.warning("Please fill all input fields")

        else:

            features = np.array([[longitude, latitude, housing_age,
                                total_rooms, total_bedrooms,
                                population, households, median_income,
                                INLAND, NEAR_BAY, NEAR_OCEAN, ISLAND, LESS_1H_OCEAN]])

            prediction = model.predict(features)[0]

            st.success(f"Predicted Price: ${prediction:,.2f}")

        report = pd.DataFrame({
    "Longitude": [longitude],
    "Latitude": [latitude],
    "Housing Age": [housing_age],
    "Total Rooms": [total_rooms],
    "Total Bedrooms": [total_bedrooms],
    "Population": [population],
    "Households": [households],
    "Median Income": [median_income],
    "Ocean Proximity": [ocean],
    "Predicted House Price": [prediction]
})


        st.download_button(
            label="📥 Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="house_price_prediction.csv",
            mime="text/csv"
        )
        # -------------------------
        # PREDICTION GAUGE
        # -------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted House Price"},
            gauge={
                'axis': {'range': [0, 1000000]},
                'bar': {'color': "lightgreen"},
                'steps': [
                    {'range': [0, 300000], 'color': "red"},
                    {'range': [300000, 600000], 'color': "orange"},
                    {'range': [600000, 1000000], 'color': "green"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
with col2:

    st.subheader("Feature Importance")

    feature_names = [
        "Longitude","Latitude","Housing Age","Rooms",
        "Bedrooms","Population","Households","Income",
        "INLAND","NEAR BAY","NEAR OCEAN","ISLAND","<1H OCEAN"
    ]

    importance = model.coef_

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance")

    fig2 = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation='h',
        color="Importance",
        color_continuous_scale="viridis"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# DATASET VISUALIZATION
# -------------------------
st.divider()
st.subheader("Dataset Insights")

# sample dataset visualization
data = pd.read_csv("data/housing.csv")

col3, col4 = st.columns(2)

with col3:

    st.write("Median Income vs House Price")

    fig3 = px.scatter(
        data,
        x="median_income",
        y="median_house_value",
        color="median_house_value",
        color_continuous_scale="plasma"
    )

    st.plotly_chart(fig3, use_container_width=True)

with col4:

    st.write("House Price Distribution")

    fig4 = px.histogram(
        data,
        x="median_house_value",
        nbins=50,
        color_discrete_sequence=["cyan"]
    )

    st.plotly_chart(fig4, use_container_width=True)