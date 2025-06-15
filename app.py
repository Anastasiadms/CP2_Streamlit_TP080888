import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import hashlib
import time
import matplotlib.pyplot as plt

# Setup for Blockchain

class Block:
    def __init__(self, index, timestamp, data, prev_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.timestamp}{self.data}{self.prev_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "Genesis Block", "0")
        self.chain.append(genesis_block)

    def add_block(self, data):
        last_block = self.chain[-1]
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            data=data,
            prev_hash=last_block.hash
        )
        self.chain.append(new_block)

fraud_chain = Blockchain()


# App Title


st.title("Integrating Hybrid Model Fraud Detection with Blockchain for Detecting Online Payment Fraud")

uploaded_file = st.file_uploader("Upload the transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())


    # Load the Hybrid Model and Params

    rf_model = joblib.load("rf_model.pkl")
    ocsvm_model = joblib.load("ocsvm_model.pkl")
    with open("ga_hybrid_weights.pkl", "rb") as f:
        hybrid_params = pickle.load(f)

    weights = hybrid_params["weights"]
    threshold = hybrid_params["threshold"]

    # Prediction

    rf_proba = rf_model.predict_proba(df)[:, 1]
    ocsvm_raw = ocsvm_model.predict(df)
    ocsvm_preds = np.where(ocsvm_raw == -1, 1, 0)

    hybrid_score = weights[0] * rf_proba + weights[1] * ocsvm_preds
    hybrid_preds = (hybrid_score > threshold).astype(int)

    df["predicted_fraud"] = hybrid_preds
    st.write("### Predictions", df[["predicted_fraud"]].value_counts().rename("count"))

    # SHAP Explanation

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(df)

    st.write("### SHAP Feature Importance (Top 10)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], df, plot_type="bar", show=False)
    st.pyplot(fig)

    # Force Plot (First Fraud)

    frauds = df[df["predicted_fraud"] == 1]
    if not frauds.empty:
        idx = frauds.index[0]
        st.write("### SHAP Force Plot for First Predicted Fraud")
        shap.initjs()
        st_shap = shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][idx],
            df.iloc[idx],
            matplotlib=True,
        )
        st.pyplot(bbox_inches='tight')
    else:
        st.info("No fraudulent transactions predicted.")

    # Blockchain Logging

    st.write("### Blockchain Fraud Log")
    for idx, row in df[df["predicted_fraud"] == 1].iterrows():
        data = {
            "client_id": row.get("client_id", idx),
            "amount": row.get("amount", 0),
            "predicted_fraud": int(row["predicted_fraud"]),
            "origin_balance": row.get("oldbalanceOrg", 0),
            "destination_balance": row.get("newbalanceDest", 0),
            "transaction_type": row.get("type_code", "N/A"),
        }
        fraud_chain.add_block(data)

    for block in fraud_chain.chain:
        st.write(f"**Block {block.index}**")
        st.json({
            "timestamp": time.ctime(block.timestamp),
            "data": block.data,
            "hash": block.hash,
            "prev_hash": block.prev_hash
        })
        st.markdown("---")
