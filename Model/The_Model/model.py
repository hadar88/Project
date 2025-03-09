import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from make_dataset import MenusDataset, read_foods_tensor, FoodProperties as FP
from menu_output_transform import transform2
import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SPLIT = ["train", "val", "test"][0]

MODEL_VERSION = 1.0
BATCH_SIZE = 256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="The split to use (train, val, test)", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load the dataset ###

    split = SPLIT if args.split is None else args.split

    print(f"Loading {split} set...")
    menus = MenusDataset(split=SPLIT)
    # menus = Subset(menus, range(10))
    dataloader = DataLoader(menus, batch_size=BATCH_SIZE, shuffle=(SPLIT == "train"))
    
    model = MenuGenerator().to(device)

    if split == "train":
        ### Define the model, loss function and optimizer ###

        myLoss = MenuLoss(device).to(device)                                       
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

        ### Train and save the model ###

        print("Training...")
        train_model(dataloader, model, myLoss, optimizer, 30, device, True)

        torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}.pth")
        print(f"Model saved as saved_models/model_v{MODEL_VERSION}.pth")

        evaluate_on_random_sample(dataloader, model, device)

    elif split == "val" or split == "test":
        ### Load the model and evaluate it ###

        model.load_state_dict(torch.load(f"saved_models/model_v{MODEL_VERSION}.pth", weights_only=True))
        model.eval()

        evaluate_on_random_sample(dataloader, model, device)

    loss = evaluate_model(dataloader, model, MenuLoss(device), device)
    print(f"Loss on the {split} set: {loss:.4f}")

class MenuGenerator(nn.Module):
    def __init__(self):
        super(MenuGenerator, self).__init__()

        # TODO: change this to a more suitable architecture

        self.fc1 = nn.Linear(14, 210)   # 14 is the number of features in the input (Calories, Carb, ...)
        self.fc2 = nn.Linear(210, 210)
        self.fc3 = nn.Linear(210, 420)

    def forward(self, x):
        y = torch.relu(self.fc1(x))

        y = torch.relu(self.fc2(y))

        y = torch.relu(self.fc3(y))

        y = y.reshape(-1, 7, 3, 10, 2)
        
        return y

class MenuLoss(nn.Module):
    def __init__(self, device):
        super(MenuLoss, self).__init__()

        self.ZERO_NONZERO_PENALTY = 3
        self.OUT_OF_RANGE_PENALTY = 2
        self.NUTRITION_PENALTY = 2
        self.PREF_PENALTY = 2
        self.ALERGENS_PENALTY = 3
        self.INGREDIENTS_PENALTY = 1
        self.MEAL_PENALTY = 1
        self.MSE_PENALTY = 1
        
        self.device = device

        self.data = read_foods_tensor().to(device)

    def forward(self, y_pred, y):
        l1loss = nn.L1Loss()
        mseloss = nn.MSELoss()

        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]

        true_ids = y[..., 0]
        true_amounts = y[..., 1]

        ### Penalize the model for giving rows with no id but with an amount or vice versa ###
        
        zero_id = self.zero_mask(pred_ids)
        zero_amount = self.zero_mask(pred_amounts)

        zero_nonzero_penalty = l1loss(zero_id, zero_amount) # TODO: fix without l1loss

        ### Penalize the model for predicting an out-of-range IDs but keeping good diversity of the IDs ###

        id_out_of_range = torch.relu(pred_ids - 222)
        id_out_of_range = 1 - self.zero_mask(id_out_of_range)
        id_range_penalty =  id_out_of_range.sum(dim=(1, 2, 3)).mean()

        ### Compute the difference between y_pred and y ###

        nutrition_diff = 0.0    # Calories, Carbs, Sugars, Fat, Protein
        preferences_diff = 0.0  # Vegetarian, Vegan
        alergens_diff = 0.0     # Contains Eggs, Gluten, Milk, Peanuts, Soy, Fish, Sesame
        ingredients_diff = 0.0  # Fruit, Vegetable, Cheese, Meat, Cereal

        for fp in [FP.CALORIES, FP.CARBOHYDRATE, FP.SUGARS, FP.FAT, FP.PROTEIN]:
            gold = (self.get_continuous_value(true_ids, fp) * true_amounts / 100).sum(dim=(1, 2, 3)) / 7
            pred = (self.get_continuous_value(self.round_and_bound(pred_ids), fp) * pred_amounts / 100).sum(dim=(1, 2, 3)) / 7
            nutrition_diff += l1loss(pred, gold) / 100  # TODO: fix without l1loss
            
        for fp in [FP.VEGETARIAN, FP.VEGAN]:
            gold = (1 - self.get_binary_value(true_ids, fp)).sum(dim=(1, 2, 3))
            pred = (1 - self.get_binary_value(self.round_and_bound(pred_ids), fp)).sum(dim=(1, 2, 3))
            preferences_diff += (torch.exp(-10 * gold) * pred.pow(2)).mean()

        for fp in [FP.CONTAINS_EGGS, FP.CONTAINS_GLUTEN, FP.CONTAINS_MILK, FP.CONTAINS_PEANUTS_OR_NUTS, FP.CONTAINS_SOY, FP.CONTAINS_FISH, FP.CONTAINS_SESAME]:
            gold = self.get_binary_value(true_ids, fp).sum(dim=(1, 2, 3))
            pred = self.get_binary_value(self.round_and_bound(pred_ids), fp).sum(dim=(1, 2, 3))
            alergens_diff += (torch.exp(-10 * gold) * pred.pow(2)).mean()

        for fp in [FP.FRUIT, FP.VEGETABLE, FP.CHEESE, FP.MEAT, FP.CEREAL]:
            gold = self.get_binary_value(true_ids, fp).sum(dim=(1, 2, 3))
            pred = self.get_binary_value(self.round_and_bound(pred_ids), fp).sum(dim=(1, 2, 3))
            ingredients_diff += l1loss(pred, gold) / 100    # TODO: fix without l1loss

        ### Compute differences in calories in breakfast, lunch and dinner ###

        meals_diff = 0.0

        for i in range(3):
            gold = (self.get_continuous_value(true_ids[:, :, i], FP.CALORIES) * true_amounts[:, :, i] / 100).sum(dim=(1, 2)) / 7
            pred = (self.get_continuous_value(self.round_and_bound(pred_ids[:, :, i]), FP.CALORIES) * pred_amounts[:, :, i] / 100).sum(dim=(1, 2)) / 7
            meals_diff += l1loss(pred, gold) / 100

        ### Compute the MSE ###

        pred_calorie_value = self.get_continuous_value(self.round_and_bound(pred_ids), FP.CALORIES) / 100
        pred_calories_per_day = (pred_amounts * pred_calorie_value).sum(dim=(2, 3))
        pred_mses = ((pred_calories_per_day - pred_calories_per_day.mean(dim=1, keepdim=True)).pow(2)).mean(dim=1)

        ### Compute the total loss ###

        loss = self.ZERO_NONZERO_PENALTY * zero_nonzero_penalty
        loss += self.OUT_OF_RANGE_PENALTY * id_range_penalty
        loss += 100 * self.entropy_penalty(pred_ids)

        loss += self.NUTRITION_PENALTY * nutrition_diff
        loss += self.PREF_PENALTY * preferences_diff
        loss += self.ALERGENS_PENALTY * alergens_diff
        loss += self.INGREDIENTS_PENALTY * ingredients_diff
        loss += self.MEAL_PENALTY * meals_diff
        loss += self.MSE_PENALTY * pred_mses.mean()

        return loss
    
    def entropy_penalty(self, pred_ids):
        # Apply softmax to the predicted logits (IDs)
        softmax_probs = torch.nn.functional.softmax(pred_ids, dim=-1)
        
        # Compute entropy (the higher the entropy, the more uniform the distribution)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=-1).mean()
        
        return entropy

    def get_binary_value(self, x, category: FP):
        return torch.sum(
            torch.stack([
                    v * torch.exp(-((x - i * v).pow(2)) / 0.01)
                    for i, v in enumerate(self.data[:, category.value])],
                dim=0,
            ),
            dim=0,
        )

    def get_continuous_value(self, x, category: FP):
        return torch.sum(
            torch.stack(
                [
                    v * torch.exp(-((x - i).pow(2)) / 0.01)
                    for i, v in enumerate(self.data[:, category.value])
                ],
                dim=0,
            ),
            dim=0,
        )

    class RoundSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return torch.round(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def round_ste(self, input):
        return self.RoundSTE.apply(input)

    def bound(self, x):
        """approx. 0 for any id > 222 and the id itself for any id <= 222."""
        return x * torch.sigmoid(50 * (222.5 - x))

    def round_and_bound(self, x):
        return self.bound(self.round_ste(x))
    
    def zero_mask(self, x):
        return torch.exp(-4 * x)
    

def train_model(dataloader, model, criterion, optimizer, epochs, device, plot_loss=True):
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

            loss = criterion(y_pred, y)
            
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")

        loss_history.append(epoch_loss)

    plt.savefig("loss_plot.png")

    if plot_loss:
        plt.plot(loss_history)
        plt.show()

def evaluate_model(dataloader, model, criterion, device):
    """
    Evaluate the model on the given dataset.

    Returns:
        float: the average loss on the dataset.
    """

    model.eval()

    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = criterion(y_pred, y)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def evaluate_on_random_sample(dataloader, model, device):
    print("Here is a random prediction:")

    print("Reading the foods data...\n")
    FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y = dataloader.dataset[random_index]
    y_pred = model(x.unsqueeze(0).to(device))
        
    # print("For the following input:")
    # print(x)
    print()
    print("The model predicted:")
    print(y_pred.squeeze())
    # print("The ground truth was:")
    # print(y)
    # print()

    print("Here's a comparison between the ground truth and the model's prediction:")
    # print("Ground truth:")
    # print(transform2(y, data, device))
    print("Model's prediction:")
    print(transform2(y_pred.squeeze(), data, device))

if __name__ == "__main__":
    main()
