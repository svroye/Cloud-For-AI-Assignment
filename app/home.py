import requests
import streamlit as st
import logging

base_url = "http://api:8000"

logging.basicConfig(level=logging.INFO)

st.sidebar.title("Welcome!")
file = st.sidebar.file_uploader("Select a picture", type=['jpg', 'png'])


def api_call():
    response = requests.post(url=f"{base_url}/predict")
    if response.status_code == 200:
        return response.json()
    else:
        return response.text


logging.info(file)
if file:
    print("file here")
    response = api_call()
    logging.info(response)
