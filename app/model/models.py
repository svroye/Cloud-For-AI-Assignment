import pickle
from abc import ABC, abstractmethod
from ultralytics import YOLO
import torch


class Model(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, data):
        pass


class YoloModel(Model):

    def __init__(self, model):
        super().__init__(YOLO(model))

    def predict(self, data):
        predict = self.model.predict(data)
        names = predict[0].names
        probs = predict[0].probs
        prediction = names[probs.top1]
        confidence = probs.numpy().top1conf * 100
        result = {"prediction": prediction, "confidence": confidence}
        return result


class DensenetModel(Model):
    def __init__(self, model):
        with open(model, 'rb') as f:
            m = pickle.load(f)
        super().__init__(m)

    def predict(self, data):
        model = self.model["model"]
        model.eval()

        device = torch.device("cpu")
        data_tf = self.model["transformation"](data).to(device)
        data_tf = torch.unsqueeze(data_tf, 0)

        probs = torch.nn.functional.softmax(model(data_tf), dim=1)
        prediction_idx = torch.argmax(probs, dim=1).item()
        prediction_label = self.model["labels"][prediction_idx]
        predicted_probability = probs[0, prediction_idx].item()

        return {"prediction": prediction_label, "confidence": predicted_probability}


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, data):
        predictions = [model.predict(data) for model in self.models]
        return predictions
        # if all(pred["prediction"] == predictions[0]["prediction"] for pred in predictions):
        #     # If all labels are the same, return a single prediction
        #     return predictions[0]
        # else:
        #     # If labels are different, return an array of predictions
        #     return predictions


