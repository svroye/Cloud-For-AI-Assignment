import streamlit as st

def gallery():
    import streamlit as st

    st.markdown(f"# {list(page_names_to_funcs.keys())[0]}")
    st.write("Images in the gallery")


def image_classifier():
    import streamlit as st
    import logging
    from PIL import Image
    from io import BytesIO
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO)

    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.write("Within this page, you can upload an image and use the 'predict' function to evaluate what kind of sports ball is in the image.")
    st.write("Choose an image to upload to the gallery")

    file = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'], label_visibility="collapsed")

    if file is not None:
        image_bytes = file.getvalue()
        image = Image.open(BytesIO(image_bytes))
        st.image(image)
        prediction = ""
        confidence = ""

        if st.button("Predict"):
            # temp method via streamlit instead of fastapi
            model = YOLO("./last.pt")
            result = model.predict(Image.open(BytesIO(image_bytes)))
            names_dict = result[0].names
            probs = result[0].probs
            prediction = names_dict[probs.top1]
            prediction = prediction.replace("_", " ").title()
            confidence = probs.numpy().top1conf * 100

        if prediction != "" and confidence != "":
            st.write("Prediction:", prediction)
            st.write("Confidence: {:0.2f}%".format(confidence))

            manual = st.selectbox("If the prediction is incorrect, please pick the correct one",
                                  ["", "American Football", "Baseball", "Basketball", "Billiard Ball",
                                   "Bowling Ball",
                                   "Cricket Ball", "Football", "Golf Ball",
                                   "Hockey Ball", "Hockey Puck", "Rugby Ball", "Shuttlecock",
                                   "Table Tennis Ball",
                                   "Tennis Ball", "Volleyball"])
            st.write("You selected:", manual)

            # TBC via fastapi
            # result = api_call(file)

           # if st.button("Save Image"):
           #     if manual != "":
           #         label = manual.replace(" ", "_").lower()
           #     else:
           #         label = prediction

           # logging.info(response)


def api_call(file):
    import requests

    base_url = "http://api:8000"
    response = requests.post(url=f"{base_url}/predict/", data=file)

    if response.status_code == 200:
        return response.json()
    else:
        return response.text


page_names_to_funcs = {
    "Gallery": gallery,
    "Image Classifier": image_classifier
}

st.sidebar.title("Sports Balls Exhibition")
st.sidebar.write("Within this application, a user can view the gallery of sports balls that are ")

pages = st.sidebar.selectbox("Pages", page_names_to_funcs.keys())
page_names_to_funcs[pages]()
