import pickle
from abc import ABC, abstractmethod
from typing import List

from ultralytics import YOLO
import torch


class ModelPrediction:
    def __init__(self, prediction, probability):
        self.prediction = prediction
        self.probability = probability


class EnsemblePrediction:
    def __init__(self, unique_result: bool, result: List[ModelPrediction]):
        self.unique_result = unique_result
        self.result = result


class Model(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, data) -> ModelPrediction:
        pass


class YoloModel(Model):

    def __init__(self, model):
        super().__init__(YOLO(model))

    def predict(self, data):
        predict = self.model.predict(data)
        names = predict[0].names
        probs = predict[0].probs
        prediction = names[probs.top1]
        probability = probs.numpy().top1conf * 100
        return ModelPrediction(prediction, probability)


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
        prediction = self.model["labels"][prediction_idx]
        probability = probs[0, prediction_idx].item() * 100

        return ModelPrediction(prediction, probability)


class EnsembleModel:
    def __init__(self, models: List[Model]):
        self.models = models

    def predict(self, data):
        predictions = [model.predict(data) for model in self.models]

        # Check if all predicted labels are the same
        if all(pred.prediction == predictions[0].prediction for pred in predictions):
            avg_probability = sum(pred.probability for pred in predictions) / len(predictions)
            output = ModelPrediction(prediction=predictions[0].prediction, probability=avg_probability)
            return EnsemblePrediction(unique_result=True, result=[output])
        else:
            # If labels are different, sort the predictions by decreasing order of probability
            sorted_predictions = sorted(predictions, key=lambda x: x.probability, reverse=True)
            return EnsemblePrediction(unique_result=False, result=sorted_predictions)

