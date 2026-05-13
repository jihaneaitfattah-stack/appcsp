import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# CONFIG PAGE
# ==================================================
st.set_page_config(page_title="AI Energy Predictor", layout="wide")

st.title("⚡ AI Energy Prediction System")
st.write("Upload dataset, train model, and predict automatically")

# ==================================================
# UPLOAD DATASET TRAINING
# ==================================================
train_file = st.file_uploader("Upload dataset WITH target (Pelectrique)", type=["xlsx"])

if train_file:

    df = pd.read_excel(train_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = "Pelectrique"

    if target not in df.columns:
        st.error("Target column not found!")
    else:

        # =========================
        # SPLIT DATA
        # =========================
        X = df.drop(columns=[target])
        y = df[target]

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # =========================
        # TRAIN MODEL
        # =========================
        if st.button("🚀 Train Models"):

            st.info("Training Random Forest...")

            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # =========================
            # METRICS
            # =========================
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            st.success("Training completed!")

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", round(mae, 2))
            col2.metric("RMSE", round(rmse, 2))
            col3.metric("R²", round(r2, 4))

            # =========================
            # PLOT
            # =========================
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label="Real")
            ax.plot(preds, label="Prediction")
            ax.legend()
            st.pyplot(fig)

            # =========================
            # SAVE MODEL
            # =========================
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/random_forest.pkl")

            st.success("Model saved in /models")

# ==================================================
# PREDICTION PHASE
# ==================================================
st.markdown("---")
st.subheader("🔮 Prediction on new data")

test_file = st.file_uploader("Upload NEW dataset WITHOUT target", type=["xlsx"], key="test")

if test_file:

    df_new = pd.read_excel(test_file)

    st.dataframe(df_new.head())

    if st.button("Predict"):

        model = joblib.load("models/random_forest.pkl")

        df_new = df_new.fillna(0)

        predictions = model.predict(df_new)

        df_new["Prediction_Pelectrique"] = predictions

        st.success("Prediction completed!")

        st.dataframe(df_new)

        # download CSV
        csv = df_new.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download results",
            csv,
            "predictions.csv",
            "text/csv"
        )

        # plot
        st.line_chart(df_new["Prediction_Pelectrique"])
