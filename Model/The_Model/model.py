from make_dataset import MenusDataset
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 64

training_set = MenusDataset(train=True)
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = MenusDataset(train=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


