from fastapi import FastAPI
from pickle import load
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI()
model = YOLO('./last.pt')

@app.post("/predict/")
def predictor(file):
    from io import BytesIO
    # with open("last.pt", "rb") as f:
    #     model = load(f)
    # model = YOLO('./api/last.pt')

    image = Image.open(BytesIO(file["data"]))
    result = model.predict(image)

    names_dict = result[0].names
    probs = result[0].probs
    prediction = names_dict[np.argmax(probs.top1)]
    prediction = prediction.replace("_", " ").title()
    confidence = probs.top1conf * 100

    return {"prediction": prediction, "confidence": confidence}
