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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


BATCH_SIZE = 64
FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"


def read_foods_tensor():
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)
    foods.close()

    data_tensor = torch.zeros(len(data) + 1, len(data["1"]) - 1, dtype=torch.float32)

    for food_id in data:
        index = int(food_id)

        if index == 0:
            continue

        data_tensor[index][0] = data[food_id]["Calories"]
        data_tensor[index][1] = data[food_id]["Carbohydrate"]
        data_tensor[index][2] = data[food_id]["Sugars"]
        data_tensor[index][3] = data[food_id]["Fat"]
        data_tensor[index][4] = data[food_id]["Protein"]
        data_tensor[index][5] = data[food_id]["Vegetarian"]
        data_tensor[index][6] = data[food_id]["Vegan"]
        data_tensor[index][7] = data[food_id]["Contains eggs"]
        data_tensor[index][8] = data[food_id]["Contains milk"]
        data_tensor[index][9] = data[food_id]["Contains peanuts or nuts"]
        data_tensor[index][10] = data[food_id]["Contains fish"]
        data_tensor[index][11] = data[food_id]["Contains sesame"]
        data_tensor[index][12] = data[food_id]["Contains soy"]
        data_tensor[index][13] = data[food_id]["Contains gluten"]
        data_tensor[index][14] = data[food_id]["Fruit"]
        data_tensor[index][15] = data[food_id]["Vegetable"]
        data_tensor[index][16] = data[food_id]["Cheese"]
        data_tensor[index][17] = data[food_id]["Meat"]
        data_tensor[index][18] = data[food_id]["Cereal"]

    return data_tensor

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
        y = y.reshape(-1, 7, 3, 10, 2)

        # y = torch.round(y)
        
        return y

###### Loss ##########

class MenuLoss(nn.Module):
    def __init__(self):
        super(MenuLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 1.0
        self.HIGHEST_ID = 222

    def forward(self, y_pred, y):
        # Penalize the model for predicting zeros when the amount is non-zero or vice versa

        pred_ids = y_pred[..., 0]  # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]  # Shape: (batch_size, 7, 3, M)

        true_ids = y[..., 0]  # Shape: (batch_size, 7, 3, M)
        true_amounts = y[..., 1]  # Shape: (batch_size, 7, 3, M)

        ### PRED ONLY loss ###

        # For (ids == 0) * (amounts != 0)
        id_zero_mask = 1 - torch.tanh(4 * pred_ids)
        amount_nonzero_mask = torch.tanh(4 * pred_amounts)  
        case1 = id_zero_mask * amount_nonzero_mask
        
        # For (ids != 0) * (amounts == 0)
        id_nonzero_mask = 1 - id_zero_mask  
        amount_zero_mask = 1 - amount_nonzero_mask 
        case2 = id_nonzero_mask * amount_zero_mask
        
        # Combine both cases
        zeros_nonzeros = self.ZERO_NONZERO_PENALTY * (case1 + case2)

        # Penalize the model for predicting an id that is not in the dataset
        
        id_range_mask = F.relu(pred_ids - self.HIGHEST_ID)

        zeros_nonzeros = zeros_nonzeros.sum(dim=(1, 2, 3)).mean()
        id_range = id_range_mask.sum(dim=(1, 2, 3)).mean()

        ### COMPARE TO TRUE loss ###

        # Calculate the calories in both menus

        calories_for_true_ids = data[true_ids.flatten().long(), 0].reshape(y.size(0), -1)

        true_amounts = true_amounts.reshape(y.size(0), -1)

        calories_in_true_menu = (calories_for_true_ids * true_amounts / 100).sum(dim=1) / 7

        # ---

        valid_ids_mask = (pred_ids > 0) & (pred_ids <= self.HIGHEST_ID)

        pred_ids_clone = pred_ids.clone()

        pred_ids_clone[~valid_ids_mask] = 0

        calories_for_pred_ids = data[pred_ids_clone.flatten().long(), 0].reshape(y_pred.size(0), -1)

        pred_amounts = pred_amounts.reshape(y_pred.size(0), -1)

        calories_in_pred_menu = (calories_for_pred_ids * pred_amounts / 100).sum(dim=1) / 7

        calories_diff = ((calories_in_true_menu - calories_in_pred_menu) ** 2).mean()

        loss = zeros_nonzeros + id_range + calories_diff

        return loss


######################

def train_model(dataloader, model, criterion, optimizer, epochs, device):
    model.train()

    epochs_bar = tqdm(range(epochs))
    loss_history = []

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
        loss_history.append(epoch_loss)

    plt.plot(loss_history)
    # plt.savefig('loss_plot.png')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Foods Data...")
data = read_foods_tensor().to(device)

model = MenuGenerator().to(device)
# criterion = nn.MSELoss()
myLoss = MenuLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

print("Training...")

x, _ = training_set[0]

y_pred = model(x.to(device))

#print(y_pred)

train_model(training_loader, model, myLoss, optimizer, 100, device)


# y_pred = model(x.to(device))

# print(y_pred)
