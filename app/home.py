import streamlit as st

st.sidebar.title("Welcome!")
file = st.sidebar.file_uploader("Select a picture", type=['jpg', 'png'])
