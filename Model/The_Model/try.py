import torch
import numpy as np
from make_dataset import read_foods_tensor, FoodProperties as FP

# def chebyshev_series(x, coeffs):
#     N = len(coeffs)
#     if N == 1:
#         return coeffs[0] * torch.ones_like(x)

#     b_next = torch.zeros_like(x)  
#     b_curr = torch.zeros_like(x)  
    
#     for c in reversed(coeffs[1:]): 
#         b_next, b_curr = b_curr, 2 * x * b_curr - b_next + c
    
#     return x * b_curr - b_next + coeffs[0]  

def chebyshev_function(x, coefficients, domain):
    a, b = domain
    x_mapped = (2 * x - (a + b)) / (b - a)

    y = np.polynomial.chebyshev.chebval(x_mapped, coefficients)
    return y

data = read_foods_tensor()
x_data = torch.tensor(range(len(data))).numpy()
y_data = data[:, FP.FAT.value].numpy()

x_min, x_max = x_data.min(), x_data.max()
x_scaled = 2 * (x_data - x_min) / (x_max - x_min) - 1  

cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_scaled, y_data, 446)

cheb_coeffs = torch.tensor(cheb_poly.coef, dtype=torch.float32)  
domain = (x_min, x_max)
# print("Chebyshev coefficients:", cheb_coeffs)

x = torch.tensor([1, 60, 120, 222], dtype=torch.float32, requires_grad=True)
y = chebyshev_function(x, cheb_coeffs, domain)


print("Predicted y:", y)

