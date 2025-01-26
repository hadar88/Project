import json
import torch

def make_xs(dataset_file_path: str):
    xs = []

    with open(dataset_file_path, "r") as dataset_file:
        dataset = json.load(dataset_file)

        for menu_id in dataset:
            x = []

            for entry in dataset[menu_id]["Initial"]:
                x.append(dataset[menu_id]["Initial"][entry])

            xs.append(x)

        return torch.tensor(xs)


def make_labels(dataset_file_path: str):
    labels = []

    with open(dataset_file_path, "r") as dataset_file:
        dataset = json.load(dataset_file)

        for menu_id in dataset:
            label = []

            for entry in dataset[menu_id]["Menu"]:
                label.append(dataset[menu_id]["Menu"][entry])

            labels.append(label)

        return torch.tensor(labels)
    
print(make_labels("../../Data/layouts/MenusInput.json"))