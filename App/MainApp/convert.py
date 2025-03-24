# import torch

# PATH = "../../Model/The_Model/saved_model/model_v1.0.pth"

# model = torch.load(PATH, map_location=torch.device('cpu'))

# dummy_input = torch.randn(1, 14)  
# torch.onnx.export(model, dummy_input, "model.onnx", export_params=True,
#                   input_names=['input'], output_names=['output'])