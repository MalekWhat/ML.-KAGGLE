from dataset import SunCoordDataset
import torchvision.transforms.v2 as tfs
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train = SunCoordDataset("D:\Git\RepoMALEKWHAT\SunCoordCNN\data\dataset_reg", transform=transforms)

train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"),#чтобы размер карт признаков совпадал и сходным изображ.
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(), #батч не трогает
    nn.Linear(4096, 128), #128 нейронов
    nn.ReLU(),
    nn.Linear(128, 2)
)

optimizer = optim.Adam(params=model.parameters(), lr=0.002, weight_decay=0.001)
loss_function = nn.MSELoss()

epoch = 5
model.train()

for e in range(epoch):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{e + 1}/{epoch}], loss_mean={loss_mean:.3f}")

params_tr_model = model.state_dict()
torch.save(params_tr_model, "D:\Git\RepoMALEKWHAT\SunCoordCNN\model\model_sun.tar")

#Test
d_test = SunCoordDataset("D:\Git\RepoMALEKWHAT\SunCoordCNN\data\dataset_reg", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)
Q = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    p = model(x_test)
    Q += loss_function(p, y_test).item()
    count += 1

Q /= count
print(Q)