import streamlit as st
from PIL import Image
from io import BytesIO


def setSessionStates():
    if "predict" not in st.session_state:
        st.session_state["predict"] = ""

    if "select" not in st.session_state:
        st.session_state["select"] = None


def deleteSessionStates():
    for key in st.session_state.keys():
        del st.session_state[key]


def onClickFunction(img):
    result = predictor(img)
    st.session_state["predict"] = result
    return result


def predictor(img):
    from ultralytics import YOLO

    model = YOLO("./app/last.pt")
    predict = model.predict(img)
    names_dict = predict[0].names
    probs = predict[0].probs
    prediction = names_dict[probs.top1]
    prediction = prediction.replace("_", " ").title()
    confidence = probs.numpy().top1conf * 100
    result = {"prediction": prediction, "confidence": confidence}
    return result


def readPrediction():
    st.write('Prediction:', st.session_state["predict"]["prediction"])
    st.write('Confidence: {:0.2f}%'.format(st.session_state["predict"]["confidence"]))


def getManualSelection():
    selection = st.selectbox("If the prediction is incorrect, please pick the correct one",
                             ["American Football", "Baseball", "Basketball", "Billiard Ball",
                              "Bowling Ball", "Cricket Ball", "Football", "Golf Ball",
                              "Hockey Ball", "Hockey Puck", "Rugby Ball", "Shuttlecock",
                              "Table Tennis Ball", "Tennis Ball", "Volleyball"], index= None,
                             placeholder="Do not use in case of a correct prediction",)
    st.session_state.select = selection
    return st.session_state.select


def getLabel():
    if st.session_state["select"] is None:
        lbl = st.session_state["predict"]
    else:
        lbl = st.session_state["select"].replace(" ", "_").lower()
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

file = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'], label_visibility="collapsed")

if file is not None:
    image_bytes = file.getvalue()
    image = Image.open(BytesIO(image_bytes))
    image.thumbnail((256, 256))
    st.image(image)

    setSessionStates()

    if st.button("Predict"):
        onClickFunction(image)

    if st.session_state["predict"] != "":
        readPrediction()
        getManualSelection()

        if (st.session_state["select"] is not None):
            st.write("You have selected:", st.session_state["select"])

        if st.button("Save Image"):
            label = getLabel()

            # Write image with label to DB
            # saveImage(image_bytes, label)

            deleteSessionStates()
