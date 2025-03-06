from make_dataset import MenusDataset, read_foods_tensor, FoodProperties as FP
from menu_output_transform import transform_batch, transform_batch2
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from make_dataset import read_foods_tensor, FoodProperties as FP
import torch


BATCH_SIZE = 256
FOODS_DATA_PATH = "../../Data/layouts/FoodsByID.json"


if __name__ == "__main__":
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

        return y


###### Loss ##########


class MenuLoss(nn.Module):
    def __init__(self, device):
        super(MenuLoss, self).__init__()
        self.ZERO_PENALTY = 5000.0
        self.PREF_PENALTY = 100.0
        self.device = device

        self.data = read_foods_tensor().to(device)

    def get_binary_value(self, x, category: FP):
        return torch.sum(
            torch.stack([
                    v * torch.exp(-((x - i * v) ** 2) / 0.01)
                    for i, v in enumerate(self.data[:, category.value])],
                dim=0,
            ),
            dim=0,
        )

    def get_continuous_value(self, x, category: FP):
        return torch.sum(
            torch.stack(
                [
                    v * torch.exp(-((x - i) ** 2) / 0.01)
                    for i, v in enumerate(self.data[:, category.value])
                ],
                dim=0,
            ),
            dim=0,
        )

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]  # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]

        true_ids = y[..., 0]
        true_amounts = y[..., 1]

        ### Punish the model if it gives a positive amount for a 0-id food, or vice versa ###

        case1 = 1 - torch.tanh(4 * pred_ids)  # ids == 0
        case2 = 1 - torch.tanh(4 * pred_amounts)  # amounts == 0

        zeros_nonzeros_penalty = self.ZERO_PENALTY * (2 * case1 + case2)  # Combine both cases
        zeros_nonzeros_penalty = zeros_nonzeros_penalty.sum(dim=(1, 2, 3)).mean()  # Sum and average for all menus in the batch

        ### Penalize the model for predicting an id that is not in the dataset ###

        id_range_penalty = F.relu(pred_ids - 222)

        id_range_penalty = id_range_penalty.sum(dim=(1, 2, 3)).mean()

        ### Compute differences in calories, carbs, sugars, fat and proteins ###

        nutrition_diff = 0.0
        preferences_diff = 0.0

        for fp in [FP.CALORIES, FP.CARBOHYDRATE, FP.SUGARS, FP.FAT, FP.PROTEIN]:
            gold = (self.get_continuous_value(true_ids, fp) * true_amounts / 100).sum(dim=(1, 2, 3)) / 7
            pred = (self.get_continuous_value(self.round_and_mask(pred_ids), fp) * pred_amounts / 100).sum(dim=(1, 2, 3)) / 7
            nutrition_diff += (((gold - pred) / 100) ** 2).mean()
            
        for fp in [FP.VEGETARIAN]:
            gold = (1 - self.get_binary_value(true_ids, fp)).sum(dim=(1, 2, 3))
            pred = (1 - self.get_binary_value(self.round_and_mask(pred_ids), fp)).sum(dim=(1, 2, 3))
            preferences_diff += self.PREF_PENALTY * (torch.exp(-10 * gold) * pred.pow(2)).mean()

        ### Compute the total loss ###

        loss = zeros_nonzeros_penalty + id_range_penalty + nutrition_diff + preferences_diff

        return loss

    class RoundSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return torch.round(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def round_ste(self, input):
        return self.RoundSTE.apply(input)

    def mask(self, x):
        """approx. 0 for any id > 222 and the id itself for any id <= 222."""
        return x * torch.sigmoid(50 * (222.5 - x))

    def round_and_mask(self, x):
        return self.mask(self.round_ste(x))

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

            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    plt.plot(loss_history)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Foods Data...")
    data = read_foods_tensor().to(device)

    model = MenuGenerator().to(device)
    myLoss = MenuLoss(device).to(device)                                       
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print("Training...")

    x, _ = training_set[8]

    train_model(training_loader, model, myLoss, optimizer, 20, device)

    y_pred = model(x.to(device))

    print(y_pred)
