# test_simple.py
import streamlit as st

st.title("Test App")
st.write("If this shows, Streamlit is working")

from transformers import pipeline
st.write("Testing transformers...")
try:
    model = pipeline("sentiment-analysis")
    result = model("I love this!")[0]
    st.write(f"Transformers OK: {result}")
except Exception as e:
    st.error(f"Transformers error: {e}")