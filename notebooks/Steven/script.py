import pickle

from PIL import Image
import torch
from torchvision import transforms

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

class_names = ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball',
               'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock', 'table_tennis_ball',
               'tennis_ball', 'volleyball']


transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

model_extended = {
    "model": model,
    "labels": class_names,
    "transformation": transform
}


with open("model_extended.pickle", "wb") as f:
    pickle.dump(model_extended, f)


