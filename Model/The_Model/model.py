from make_dataset import MenusDataset
from menu_output_transform import transform
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

BATCH_SIZE = 64

print("Loading Trainset...")
training_set = MenusDataset(train=True)
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

print("Loading Testset...")
test_set = MenusDataset(train=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


class MenuGenerator(nn.Module):
    def __init__(self):
        super(MenuGenerator, self).__init__()

        # 14 is the number of features in the input (Calories, Carb, ...)

        self.fc1 = nn.Linear(14, 210)
        self.fc2 = nn.Linear(210, 210)
        self.fc3 = nn.Linear(210, 420)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = y.reshape(-1, 7, 3, 10, 2)
        return y

###### Loss ##########

class MenuLoss(nn.Module):
    def __init__(self):
        super(MenuLoss, self).__init__()

    def forward(self, pred, actual):
        pass

######################

def train_model(dataloader, model, criterion, optimizer, epochs, device):
    model.to(device)
    model.train()

    epochs_bar = tqdm(range(epochs))

    for _ in epochs_bar:
        total_loss = 0.0
        num_batches = 0

        for x, m, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            y_pred_transformed = transform(y_pred)

            loss = criterion(y_pred_transformed, m)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        epoch_loss = total_loss / num_batches
        epochs_bar.set_postfix_str(f"Loss = {epoch_loss}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MenuGenerator()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

print("Training...")
train_model(training_loader, model, criterion, optimizer, 500, device)

x, _, _ = training_set[0]

x = x.to(device)
model.to(device)

model(x)
