import streamlit as st
from PIL import Image
from io import BytesIO
from model.models import YoloModel, DensenetModel, EnsembleModel, EnsemblePrediction
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd


class SessionStateKey(Enum):
    PREDICT = 'predict'
    SELECT = 'select'
    IMAGE_SELECTED = 'image_selected'


def set_session_state(key: SessionStateKey, value):
    st.session_state[key.value] = value


def get_session_state(key: SessionStateKey):
    return st.session_state[key.value]


def has_session_state(key: SessionStateKey):
    return key.value in st.session_state


def reset_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


def predict(img):
    models = [
        YoloModel("./app/model/last.pt"),
        DensenetModel("./app/model/model.pickle")
    ]
    ensemble = EnsembleModel(models)
    result = ensemble.predict(img)
    set_session_state(SessionStateKey.PREDICT, result)


def visualize_prediction(list_df):
    # Create a grouped bar chart
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    bar_width = 0.35
    index = range(len(list_df[0]))
    print(list_df[0])
    names = ["yolo_model", "densenet_model"]
    plt.suptitle('Top 5 Categories for yolo_model and densenet_model')
    for id in range(len(list_df)):

        bars = ax[id].bar(index, list_df[id]['Probability'], bar_width, label=names[id])

        ax[id].set_xlabel('Category')
        ax[id].set_ylabel('Probability')
        ax[id].set_xticks([i + bar_width / 2 for i in index])
        ax[id].set_xticklabels(list_df[id]['Prediction'], rotation=90)
        ax[id].legend()
        # Display exact values on top of the bars in the first subplot
        for bar in bars:
            yval = bar.get_height()
            ax[id].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 1), ha='center', va='bottom')
    # Display the chart in Streamlit
    st.pyplot(fig)


def read_prediction():
    ensemble_prediction: EnsemblePrediction = get_session_state(SessionStateKey.PREDICT)
    st.write("Number of models used: ", ensemble_prediction.number_of_models)
    list_df = []
    for model_output in ensemble_prediction.result:
        list_df.append(pd.DataFrame(model_output.top_results.items(), columns=["Prediction", "Probability"]))

    visualize_prediction(list_df)
    if ensemble_prediction.unique_result:
        avg_probability = sum(pred.probability for pred in ensemble_prediction.result) / len(ensemble_prediction.result)
        st.write('Prediction:', ensemble_prediction.result[0].prediction)
        st.write('Confidence:', "{:0.4f}%".format(avg_probability))


def get_manual_selection():
    selection = st.selectbox("If the prediction is incorrect, please pick the correct one",
                             ["American Football", "Baseball", "Basketball", "Billiard Ball",
                              "Bowling Ball", "Cricket Ball", "Football", "Golf Ball",
                              "Hockey Ball", "Hockey Puck", "Rugby Ball", "Shuttlecock",
                              "Table Tennis Ball", "Tennis Ball", "Volleyball"], index= None,
                             placeholder="Do not use in case of a correct prediction",)
    if selection:
        set_session_state(SessionStateKey.SELECT, selection)
    return selection


def get_label():
    if not has_session_state(SessionStateKey.SELECT):
        return get_session_state(SessionStateKey.PREDICT)
    else:
        return get_session_state(SessionStateKey.SELECT).replace(" ", "_").lower()


def on_image_change():
    reset_session_state()
    set_session_state(SessionStateKey.IMAGE_SELECTED, True)


def save_image(image_bytes):
    # label = getLabel()
    # Write image with label to DB
    # saveImage(image_bytes, label)
    reset_session_state()


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
st.sidebar.write("The Photographer Company specializes in capturing images of "
                 "sports balls used in 15 different sports.")
st.sidebar.write("Within this application, a photographer can upload their image to the database. "
                 "Before the image gets uploaded, we need to classify the image by giving it a label corresponding "
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
                        on_change=on_image_change)

if file is not None and has_session_state(SessionStateKey.IMAGE_SELECTED):
    image_bytes = file.getvalue()
    image = Image.open(BytesIO(image_bytes))
    image.thumbnail((256, 256))
    st.image(image)

    st.button("Predict", on_click=predict, args=(image,))

    if has_session_state(SessionStateKey.PREDICT):
        read_prediction()
        get_manual_selection()

        if has_session_state(SessionStateKey.SELECT):
            st.write("You have selected:", get_session_state(SessionStateKey.SELECT))

        st.button("Save Image", on_click=save_image, args=(image_bytes, ))
