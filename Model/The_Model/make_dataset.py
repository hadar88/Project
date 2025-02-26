import json
import torch
import menu_output_transform as mot
from torch.utils.data import Dataset

MENUS_INPUT = "../../Data/layouts/MenusInput.json"
MENUS_BY_ID = "../../Data/layouts/MenusById.json"

def make_xs():
    xs = []

    with open(MENUS_INPUT, "r") as dataset_file:
        dataset = json.load(dataset_file)

        for menu_id in dataset:
            x = []

            for entry in dataset[menu_id]["Initial"]:
                x.append(dataset[menu_id]["Initial"][entry])

            xs.append(x)

        return torch.tensor(xs)


def make_mids():
    labels = []

    with open(MENUS_INPUT, "r") as dataset_file:
        dataset = json.load(dataset_file)

        for menu_id in dataset:
            label = []

            for entry in dataset[menu_id]["Menu"]:
                label.append(dataset[menu_id]["Menu"][entry])

            labels.append(label)

        return torch.tensor(labels)
    

def make_ys():
    ys = []
    max_len = 0

    with open(MENUS_BY_ID, "r") as dataset_file:
        dataset = json.load(dataset_file)

        for menu_id in dataset:
            y = dataset[menu_id]
            y = mot.menu_dict_to_tensor(y)  # shape: 7x3xLx2
            max_len = max(max_len, y.shape[2])
            
            ys.append(y)

        for i in range(len(ys)):
            y = torch.zeros(7, 3, max_len, 2)
            y[:, :, :ys[i].shape[2], :] = ys[i]
            ys[i] = y
        
        return torch.stack(ys)
    
# The DataSet

class MenusDataset(Dataset):
    def __init__(self, train: bool = True):
        xs = make_xs()
        mids = make_mids()
        ys = make_ys()

        self.xs = xs[:int(0.8 * len(xs))] if train else xs[int(0.8 * len(xs)):]
        self.mids = mids[:int(0.8 * len(mids))] if train else mids[int(0.8 * len(mids)):]
        self.ys = ys[:int(0.8 * len(ys))] if train else ys[int(0.8 * len(ys)):]

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, index):
        return self.xs[index], self.mids[index], self.ys[index]

# mids = make_mids()
# print(mids[0])