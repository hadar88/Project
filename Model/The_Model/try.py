import torch
import numpy as np
from make_dataset import read_foods_tensor, FoodProperties as FP

def chebyshev_series(x, coeffs):
    N = len(coeffs)
    if N == 1:
        return coeffs[0] * torch.ones_like(x)

    b_next = torch.zeros_like(x)  
    b_curr = torch.zeros_like(x)  
    
    for c in reversed(coeffs[1:]): 
        b_next, b_curr = b_curr, 2 * x * b_curr - b_next + c
    
    return x * b_curr - b_next + coeffs[0]  

data = read_foods_tensor()
x_data = torch.tensor(range(len(data))).numpy()
y_data = data[:, FP.CALORIES.value].numpy()

x_min, x_max = x_data.min(), x_data.max()
x_scaled = 2 * (x_data - x_min) / (x_max - x_min) - 1  

cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_scaled, y_data, 446)

cheb_coeffs = torch.tensor(cheb_poly.coef, dtype=torch.float32)  

x = torch.tensor(222, dtype=torch.float32, requires_grad=True)

x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1  

y = chebyshev_series(x_scaled, cheb_coeffs)

# Compute gradient
y.backward()
print("Predicted y:", y.item())
print("Gradient dy/dx:", x.grad.item())
