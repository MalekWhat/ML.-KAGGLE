import torchvision.transforms.v2 as tfs
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

path_to_test = "D:/Git/RepoMALEKWHAT/SunCoordCNN/data/dataset_reg/test"
image_number = 100

st = torch.load("D:/Git/RepoMALEKWHAT/SunCoordCNN/model/model_sun.tar", weights_only=False)
model.load_state_dict(st)

with open(os.path.join(path_to_test, "format.json"), "r") as fp:
    format = json.load(fp)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
img = Image.open(os.path.join(path_to_test, f"sun_reg_{image_number}.png")).convert("RGB")
img_t = transforms(img).unsqueeze(0)

model.eval()
predict = model(img_t)
print(predict)
print(tuple(format.values())[image_number-1])
p = predict.detach().squeeze().numpy()

plt.imshow(img)
plt.scatter(p[0], p[1], s=20, c="r")
plt.show()
