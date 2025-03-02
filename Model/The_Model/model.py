from make_dataset import MenusDataset
from menu_output_transform import transform_batch, transform_batch2
import torch
import json
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"

foods = open(FOODS_DATA_PATH, "r")
data = json.load(foods)
foods.close()

print("Loading Trainset...")
training_set = MenusDataset(train=True)
# training_subset = Subset(training_set, range(1000))
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

# print("Loading Testset...")
# test_set = MenusDataset(train=False)
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


class MenuGenerator(nn.Module):
    def __init__(self):
        super(MenuGenerator, self).__init__()

        # 14 is the number of features in the input (Calories, Carb, ...)

        self.fc1 = nn.Linear(14, 210)
        self.fc2 = nn.Linear(210, 210)
        self.fc3 = nn.Linear(210, 420)

    def forward(self, x):
        y = torch.relu(self.fc1(x))
        y = torch.relu(self.fc2(y))
        y = torch.relu(self.fc3(y))
        y = torch.clamp(y, 0, 222)
        y = y.reshape(-1, 7, 3, 10, 2)
        # y = torch.round(y)
        
        return y

###### Loss ##########

class MenuLoss(nn.Module):
    def __init__(self):
        super(MenuLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 1.0

    def forward(self, y_pred, y):
        # Penalize the model for predicting zeros when the amount is non-zero or vice versa

        ids = y_pred[..., 0]  # Shape: (batch_size, 7, 3, M)
        amounts = y_pred[..., 1]  # Shape: (batch_size, 7, 3, M)

        # For (ids == 0) * (amounts != 0)
        id_zero_mask = 1 - torch.tanh(4 * ids)
        amount_nonzero_mask = torch.tanh(4 * amounts)  
        case1 = id_zero_mask * amount_nonzero_mask
        
        # For (ids != 0) * (amounts == 0)
        id_nonzero_mask = 1 - id_zero_mask  
        amount_zero_mask = 1 - amount_nonzero_mask 
        case2 = id_nonzero_mask * amount_zero_mask
        
        # Combine both cases
        zeros_nonzeros = self.ZERO_NONZERO_PENALTY * (case1 + case2)

        # Sum across dimensions and take mean across batch
        loss = zeros_nonzeros.sum(dim=(1, 2, 3))
        return loss.mean()

######################

def train_model(dataloader, model, criterion, optimizer, epochs, device):
    model.train()

    epochs_bar = tqdm(range(epochs))

    for _ in epochs_bar:
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            # y_pred_transformed = transform_batch2(y_pred, data, device)

            # loss = criterion(y_pred_transformed, m)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        epoch_loss = total_loss / num_batches
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MenuGenerator().to(device)
# criterion = nn.MSELoss()
myLoss = MenuLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


print("Training...")

x, _ = training_set[0]

y_pred = model(x.to(device))

print(y_pred)

train_model(training_loader, model, myLoss, optimizer, 200, device)


y_pred = model(x.to(device))

print(y_pred)
