
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Website title
st.title("🏠 House Price Prediction System")
st.write("Enter the house details to predict its estimated price.")

# User inputs
area = st.number_input(
    "House Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1200
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Prediction button
if st.button("Predict House Price"):

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    predicted_price = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ₹{predicted_price:,.0f}"
    )

    st.caption(
        "This is a sample ML estimate, not an actual market valuation."
    )
