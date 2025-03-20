import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from make_dataset import MenusDataset, read_foods_tensor
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
    
    model = MenuGenerator()

    if split == "train":                              
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_food_id = nn.CrossEntropyLoss()
        criterion_amount = nn.MSELoss()

        train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, optimizer, 200, device, True)

        torch.save(model.state_dict(), f"saved_models/model_v{MODEL_VERSION}.pth")
        print(f"Model saved as saved_models/model_v{MODEL_VERSION}.pth")

        evaluate_transformer_on_random_sample(dataloader, model, device)

class MenuGenerator(nn.Module):
    def __init__(self):
        super(MenuGenerator, self).__init__()

        self.emb_dim = 16

        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 256)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )

        self.food_fc = nn.Linear(256, 7 * 3 * 10 * 223)
        self.amount_fc = nn.Linear(256, 7 * 3 * 10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        x = x.unsqueeze(0) 
        x = self.transformer(x)
        x = x.squeeze(0)

        food_logits = self.food_fc(x)
        food_logits = food_logits.view(-1, 7, 3, 10, 223)

        amount = self.amount_fc(x)
        amount = amount.view(-1, 7, 3, 10, 1)
        amount = self.activation(amount)

        return food_logits, amount

def train_transformer_model(dataloader, model, criterion_food_id, criterion_amount, optimizer, epochs, device, plot_loss=True):
    model.to(device)
    model.train()

    bar = tqdm(range(epochs))

    loss_history = []

    for _ in bar:
        epoch_loss = 0.0

        for x, ids, amounts in dataloader:
            x, ids, amounts = x.to(device), ids.to(device), amounts.to(device)

            optimizer.zero_grad()

            # forward
            food_logits, pred_amounts = model(x)

            # reshape for loss computation
            food_logits = food_logits.view(-1, 223)
            ids = ids.view(-1)

            pred_amounts = pred_amounts.view(-1, 1)
            amounts = amounts.view(-1, 1)

            # compute losses
            loss_id = criterion_food_id(food_logits, ids)
            loss_amount = criterion_amount(pred_amounts, amounts)

            # joint loss weighted importance
            loss = loss_id + loss_amount
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    if plot_loss:
        plt.plot(loss_history)
        plt.savefig("loss_plot.png")
        plt.show()

def evaluate_transformer_on_random_sample(dataloader, model, device):
    model.eval()
    model.to(device)

    print("Here is a random prediction:")

    print("Reading the foods data...\n")
    FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"
    foods = open(FOODS_DATA_PATH, "r")
    data = json.load(foods)

    random_index = torch.randint(0, len(dataloader.dataset), (1,)).item()
    x, y_id, y_amount = dataloader.dataset[random_index]
    x, y_id, y_amount = x.to(device), y_id.to(device), y_amount.to(device)

    pred_id, pred_amount = model(x.unsqueeze(0).to(device))

    pred_id, pred_amount = pred_id[0], pred_amount[0]

    pred_id = torch.argmax(pred_id, dim=-1)

    pred_amount = pred_amount.squeeze(-1)
        
    print("For the following input:")
    print(x)
    print()

    print("The model predicted:")
    merged_pred = MenusDataset.merge_ids_and_amounts(pred_id, pred_amount)
    print(merged_pred)
    print()

    print("The ground truth was:")
    merged_y = MenusDataset.merge_ids_and_amounts(y_id, y_amount)
    print(merged_y)
    print()

    print("Here's a comparison between the ground truth and the model's prediction:")
    print("Model's prediction:")
    print(transform2(merged_pred, data, device))
    print("Ground truth:")
    print(transform2(merged_y, data, device))

if __name__ == "__main__":
    main()
