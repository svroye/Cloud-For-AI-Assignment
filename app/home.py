import streamlit as st
from PIL import Image
from io import BytesIO
from model.models import YoloModel, DensenetModel, EnsembleModel, EnsemblePrediction
from enum import Enum


class SessionStateKey(Enum):
    PREDICT = 'predict'
    SELECT = 'select'


def set_session_state(key: SessionStateKey, value):
    st.session_state[key.value] = value


def get_session_state(key: SessionStateKey):
    return st.session_state[key.value]


def has_session_state(key: SessionStateKey):
    return key.value in st.session_state


def deleteSessionStates():
    for key in st.session_state.keys():
        del st.session_state[key]


def reset_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


def onClickFunction(img):
    result = predictor(img)
    set_session_state(SessionStateKey.PREDICT, result)


def predictor(img):
    models = [
        YoloModel("./app/model/last.pt"),
        DensenetModel("./app/model/model.pickle")
    ]
    ensemble = EnsembleModel(models)
    return ensemble.predict(img)


def read_prediction():
    ensemble_prediction: EnsemblePrediction = get_session_state(SessionStateKey.PREDICT)
    st.write("Number of models used: ", ensemble_prediction.number_of_models)
    if ensemble_prediction.unique_result:
        model_output = ensemble_prediction.result[0]
        st.write('Prediction:', model_output.prediction)
        st.write('Confidence:', "{:0.4f}".format(model_output.probability))
    else:
        st.write("Different models returned a different result. In descending order")
        for model_output in ensemble_prediction.result:
            st.write('Prediction:', model_output.prediction)
            st.write('Confidence:', model_output.probability)


def getManualSelection():
    selection = st.selectbox("If the prediction is incorrect, please pick the correct one",
                             ["American Football", "Baseball", "Basketball", "Billiard Ball",
                              "Bowling Ball", "Cricket Ball", "Football", "Golf Ball",
                              "Hockey Ball", "Hockey Puck", "Rugby Ball", "Shuttlecock",
                              "Table Tennis Ball", "Tennis Ball", "Volleyball"], index= None,
                             placeholder="Do not use in case of a correct prediction",)
    set_session_state(SessionStateKey.SELECT, selection)
    return selection


def getLabel():
    if not has_session_state(SessionStateKey.SELECT):
        lbl = get_session_state(SessionStateKey.PREDICT)
    else:
        lbl = get_session_state(SessionStateKey.SELECT).replace(" ", "_").lower()
    return lbl


def saveImage(image_bytes, label):
    # connect with api to DB
    # pass data to DB
    return None


def api_call(file):
    import requests

    base_url = "http://api:8000"
    response = requests.post(url=f"{base_url}/predict/", files=file)

    if response.status_code == 200:
        return response.json()
    else:
        return response.text


# Sidebar
st.sidebar.title("The Photographer Company")
st.sidebar.write("The Photographer Company specializes in capturing images of"
                 "sports balls used in 15 different sports.")
st.sidebar.write("Within this application, a photographer can upload their image to the database."
                 "Before the image gets uploaded, we need to classify the image by giving it a label corresponding"
                 "to the sports ball that is visible in the image.")
st.sidebar.write("A prediction of the classification of the image will run automatically with the use of AI.")
st.sidebar.write("In case of an incorrect prediction, the photographer can manually edit the label before upload.")


# Main Page
st.markdown("# Image Classifier")
st.write(
    "Within this page, you can upload an image and use the 'predict' "
    "function to evaluate what kind of sports ball is in the image.")
st.write("Choose an image to upload to the gallery")

file = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'], label_visibility="collapsed",
                        on_change=reset_session_state)

if file is not None:
    image_bytes = file.getvalue()
    image = Image.open(BytesIO(image_bytes))
    image.thumbnail((256, 256))
    st.image(image)

    if st.button("Predict"):
        onClickFunction(image)

    if has_session_state(SessionStateKey.PREDICT):
        read_prediction()
        getManualSelection()

        if get_session_state(SessionStateKey.SELECT) is not None:
            st.write("You have selected:", get_session_state(SessionStateKey.SELECT))

        if st.button("Save Image"):
            label = getLabel()

            # Write image with label to DB
            # saveImage(image_bytes, label)

            deleteSessionStates()
