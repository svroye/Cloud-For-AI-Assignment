import logging
import pickle
from abc import ABC, abstractmethod
from typing import List

from ultralytics import YOLO
import torch


class ModelPrediction:
    def __init__(self,top_results, prediction, probability):
        self.prediction = prediction
        self.probability = probability
        self.top_results = top_results


class EnsemblePrediction:
    def __init__(self, unique_result: bool, number_of_models: int, result: List[ModelPrediction]):
        self.unique_result = unique_result
        self.number_of_models = number_of_models
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
        top_results = {
        
        }
        for idx, prob in zip(probs.top5, probs.top5conf):
            top_results[names[idx]] = prob.item() * 100
        probability = probs.numpy().top1conf * 100
        return ModelPrediction(top_results,prediction, probability)


class DensenetModel(Model):
    def __init__(self, model):
        with open(model, 'rb') as f:
            m = pickle.load(f)
        super().__init__(m)

    def predict(self, data):
        try:
            model = self.model["model"]
            model.eval()
            top_results = {}
            device = torch.device("cpu")
            data_tf = self.model["transformation"](data).to(device)
            data_tf = torch.unsqueeze(data_tf, 0)

            probs = torch.nn.functional.softmax(model(data_tf), dim=1)
            prediction_idx = torch.argmax(probs, dim=1).item()
            prediction = self.model["labels"][prediction_idx]
            probability = probs[0, prediction_idx].item() * 100
            topk_values, topk_indices = torch.topk(probs, k=5, dim=1)
            for idx, prob in zip(topk_indices.flatten(), topk_values.flatten()):
                top_results[self.model["labels"][idx.item()]] = prob.item() * 100
            return ModelPrediction(top_results, prediction, probability)
        except Exception as e:
            logging.error("Prediction could not be made. To be investigated. Exception: %s", e)
            return None


class EnsembleModel:
    def __init__(self, models: List[Model]):
        self.models = models

    def predict(self, data):
        predictions = list(filter(lambda x: x is not None, [model.predict(data) for model in self.models]))

        # Check if all predicted labels are the same
        unique_result = all(pred.prediction == predictions[0].prediction for pred in predictions)
        return EnsemblePrediction(unique_result, result=predictions, number_of_models=len(predictions))

