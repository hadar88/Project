from make_dataset import MenusDataset, read_foods_tensor, FoodProperties as FP
from menu_output_transform import transform_batch, transform_batch2
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from make_dataset import read_foods_tensor, FoodProperties as FP
import torch
from numpy.polynomial.chebyshev import Chebyshev


BATCH_SIZE = 64
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

        # y = torch.round(y)
        
        return y

###### Loss ##########

class MenuLoss(nn.Module):
    def __init__(self):
        super(MenuLoss, self).__init__()
        self.ZERO_NONZERO_PENALTY = 1.0
        self.HIGHEST_ID = 222
        
        # Compute Chebyshev coefficients
        data = read_foods_tensor()
        x = torch.tensor(range(len(data)))
        
        calories = data[:, FP.CALORIES.value]
        calories_poly = Chebyshev.fit(x, calories, 446)
        self.calories_coeffs = torch.tensor(calories_poly.coef)

    def forward(self, y_pred, y):
        pred_ids = y_pred[..., 0]       # Shape: (batch_size, 7, 3, M)
        pred_amounts = y_pred[..., 1]   

        true_ids = y[..., 0]            
        true_amounts = y[..., 1]        

        ### Punish the model if it gives a positive amount for a 0-id food, or vice versa ###

        id_zero_mask = 1 - torch.tanh(4 * pred_ids)         # ids == 0
        amount_nonzero_mask = torch.tanh(4 * pred_amounts)  # amounts != 0
        case1 = id_zero_mask * amount_nonzero_mask
        
        id_nonzero_mask = 1 - id_zero_mask          # ids != 0
        amount_zero_mask = 1 - amount_nonzero_mask  # amounts == 0
        case2 = id_nonzero_mask * amount_zero_mask
        
        zeros_nonzeros_penalty = self.ZERO_NONZERO_PENALTY * (case1 + case2)            # Combine both cases
        zeros_nonzeros_penalty = zeros_nonzeros_penalty.sum(dim=(1, 2, 3)).mean()       # Sum and average for all menus in the batch

        ### Penalize the model for predicting an id that is not in the dataset ###
        
        id_range_penalty = F.relu(pred_ids - self.HIGHEST_ID)

        id_range_penalty = id_range_penalty.sum(dim=(1, 2, 3)).mean()

        ### Compute differences in calories, carbs, sugars, fat and proteins ###

        # True:

        true_amounts = true_amounts.reshape(y.size(0), -1)  # amount of each food in y (the true "label")
        
        true_calories = self.chebyshev_eval(true_ids, self.calories_coeffs).reshape(y.size(0), -1)  # calories of each food in y (the true "label")
        calories_in_true_menu = (true_calories * true_amounts / 100).sum(dim=1) / 7

        # true_carbohydrate = data[true_ids.flatten().long(), FP.CARBOHYDRATE.value].reshape(y.size(0), -1)   # calories of each food in y (the true "label")
        # carbohydrate_in_true_menu = (true_carbohydrate * true_amounts / 100).sum(dim=1) / 7
        
        # true_sugars = data[true_ids.flatten().long(), FP.SUGARS.value].reshape(y.size(0), -1)   # calories of each food in y (the true "label")
        # sugars_in_true_menu = (true_sugars * true_amounts / 100).sum(dim=1) / 7
        
        # true_fat = data[true_ids.flatten().long(), FP.FAT.value].reshape(y.size(0), -1)   # calories of each food in y (the true "label")
        # fat_in_true_menu = (true_fat * true_amounts / 100).sum(dim=1) / 7
        
        # true_proteins = data[true_ids.flatten().long(), FP.PROTEIN.value].reshape(y.size(0), -1)   # calories of each food in y (the true "label")
        # proteins_in_true_menu = (true_proteins * true_amounts / 100).sum(dim=1) / 7

        # Pred:

        pred_amounts = pred_amounts.reshape(y_pred.size(0), -1)

        pred_calories = self.chebyshev_eval(self.round_and_mask(pred_ids), self.calories_coeffs).reshape(y_pred.size(0), -1)
        calories_in_pred_menu = (pred_calories * pred_amounts / 100).sum(dim=1) / 7
        
        # pred_carbohydrate = data[(pred_ids * valid_pred_ids_mask).flatten().long(), FP.CARBOHYDRATE.value].reshape(y_pred.size(0), -1)
        # carbohydrate_in_pred_menu = (pred_carbohydrate * pred_amounts / 100).sum(dim=1) / 7
        
        # pred_sugars = data[(pred_ids * valid_pred_ids_mask).flatten().long(), FP.SUGARS.value].reshape(y_pred.size(0), -1)
        # sugars_in_pred_menu = (pred_sugars * pred_amounts / 100).sum(dim=1) / 7
        
        # pred_fat = data[(pred_ids * valid_pred_ids_mask).flatten().long(), FP.FAT.value].reshape(y_pred.size(0), -1)
        # fat_in_pred_menu = (pred_fat * pred_amounts / 100).sum(dim=1) / 7
        
        # pred_proteins = data[(pred_ids * valid_pred_ids_mask).flatten().long(), FP.PROTEIN.value].reshape(y_pred.size(0), -1)
        # proteins_in_pred_menu = (pred_proteins * pred_amounts / 100).sum(dim=1) / 7

        # Diff:

        calories_diff = ((calories_in_true_menu - calories_in_pred_menu) ** 2).mean()
        # carbohydrate_diff = ((carbohydrate_in_true_menu - carbohydrate_in_pred_menu) ** 2).mean()
        # sugars_diff = ((sugars_in_true_menu - sugars_in_pred_menu) ** 2).mean()
        # fat_diff = ((fat_in_true_menu - fat_in_pred_menu) ** 2).mean()
        # proteins_diff = ((proteins_in_true_menu - proteins_in_pred_menu) ** 2).mean()
        
        nutrition_diff = calories_diff # + carbohydrate_diff + sugars_diff + fat_diff + proteins_diff

        ### Compute the total loss ###

        loss = zeros_nonzeros_penalty + id_range_penalty + nutrition_diff

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
        """ approx. 0 for any id > 222 and the id itself for any id <= 222. """
        return x * torch.sigmoid(50 * (222.5 - x))

    def round_and_mask(self, x):
        return self.mask(self.round_ste(x))

    def chebyshev_eval(self, x, coefficients):
        """ Evaluate a Chebyshev polynomial expansion using PyTorch. """
        # Normalize x to [-1, 1] range for Chebyshev evaluation
        x_norm = x / 111 - 1
        
        # Initialize Chebyshev polynomials
        T = [torch.ones_like(x_norm), x_norm]
        
        # Compute additional Chebyshev polynomials up to the highest degree
        for n in range(2, len(coefficients)):
            T.append(2 * x_norm * T[n-1] - T[n-2])
        
        # Compute the polynomial expansion
        result = coefficients[0] * torch.ones_like(x)
        for i in range(1, len(coefficients)):
            result += coefficients[i] * T[i]
        
        return result


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
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    plt.plot(loss_history)
    # plt.savefig('loss_plot.png')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Foods Data...")
    data = read_foods_tensor().to(device)

    model = MenuGenerator().to(device)
    myLoss = MenuLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training...")

    x, _ = training_set[0]

    y_pred = model(x.to(device))

    #print(y_pred)

    train_model(training_loader, model, myLoss, optimizer, 5, device)


    # y_pred = model(x.to(device))

    # print(y_pred)
