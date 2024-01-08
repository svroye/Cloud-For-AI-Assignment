import uvicorn
import logging
from io import BytesIO
from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI, Request, status, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()
model = YOLO('./last.pt')

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    logging.error(f"{request}: {exc_str}")
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


# class Response:
#     prediction = str
#     confidence = str


@app.post("/predict/")
def results(file: UploadFile):
    try:
        data = file.file.read()
        image = Image.open(BytesIO(data))

        predict = model.predict(image)

        names_dict = predict[0].names
        probs = predict[0].probs

        prediction = names_dict[probs.top1]
        prediction = prediction.replace("_", " ").title()

        confidence = "{:0.2f}%".format(probs.numpy().top1conf * 100)

        result = {"prediction": prediction, "confidence": confidence}
        return result
    except Exception as e:
        print(e)
    except RequestValidationError as vale:
        print(vale)


if __name__ == '__main__':
    uvicorn.run(app, host="api", port=8000)
