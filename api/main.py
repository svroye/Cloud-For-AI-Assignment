from fastapi import FastAPI
from pickle import load
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI()
with open ("last.pt", "r") as f:
    model = load(f)

#model = YOLO('./api/last.pt')


@app.post("/predict")
def predict(file):
    image = Image.frombytes(file)
    result = model(image)
    return result

@app.post("/predictor")
def predictor(file):
    # from io import BytesIO
    # # with open("last.pt", "rb") as f:
    # #     model = load(f)
    #
    # model = YOLO('./api/last.pt')
    # image = Image.open(BytesIO(file["data"]))
    # result = model.predict(image)
    #
    # names_dict = result[0].names
    # probs = result[0].probs
    # prediction = names_dict[np.argmax(probs.top1)]
    # prediction = prediction.replace("_", " ").title()
    # confidence = probs.top1conf * 100
    #
    # return {"prediction": prediction, "confidence": confidence}
    return {"data": "test"}
