import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="IAFP 2026 – Listeria in Soil", layout="wide")
st.title("Listeria in Soil — Streamlit App")

st.write("✅ App is running from `streamlit_app/app.py`")

data_dir = "data"
st.subheader("Data preview")

if os.path.exists(data_dir):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".parquet"))]
    if not files:
        st.warning("No .csv or .parquet files found in /data")
    else:
        f = st.selectbox("Choose a file", files)
        path = os.path.join(data_dir, f)

        if f.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)

        st.write("Shape:", df.shape)
        st.dataframe(df.head(200), use_container_width=True)
else:
    st.error("Folder `data/` not found in this deployment.")
