import streamlit as st
import pandas as pd
import pickle
import json

# ---------------- LOAD MODEL ----------------
with open("cat_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LOAD FEATURES ----------------
with open("top_10_features.json", "r") as f:
    FEATURES = json.load(f)

# ---------------- LOAD CLIENT DATA ----------------
data = pd.read_csv("final_data.csv")

THRESHOLD = 0.35

# ---------------- STREAMLIT UI ----------------
st.title("Bad Debt Prediction")

Customer = st.text_input("Enter Customer ID")

if st.button("Predict"):
    try:
        cust_id = int(Customer)   # convert input to int

        row = data[data["Customer"] == cust_id]

        if row.empty:
            st.error("Customer ID not found ❌")
        else:
            X = row[FEATURES]
            prob = model.predict_proba(X)[0][1]

            st.write(f"Default Probability: **{prob:.2f}**")

            if prob >= THRESHOLD:
                st.error("⚠️ Reject")
            else:
                st.success("Allow Purchase")

    except ValueError:
        st.error("Please enter a valid numeric Customer ID")
