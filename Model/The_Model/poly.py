import numpy as np
from make_dataset import read_foods_tensor, FoodProperties as FP
import torch
from numpy.polynomial.chebyshev import Chebyshev
import torch.
import matplotlib.pyplot as plt

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def round_ste(input):
    return RoundSTE.apply(input)

def mask(x):
    return x * torch.sigmoid(50 * (222.5 - x))

def compose_them(x):
    return mask(round_ste(x))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = read_foods_tensor()

x = torch.tensor(range(len(data)))

y = data[:, FP.FAT.value]

# Fit Chebyshev polynomial
coeffs = Chebyshev.fit(x, y, 446)

# Generate y values for the fitted function
y_fit = coeffs(x)

# Plot the results
plt.scatter(x, y, label="Data points", color="red")
plt.plot(x, y_fit, label="Chebyshev Fit", color="blue")
plt.legend()
plt.show()

print(coeffs.coef)

test = torch.tensor([0, 1,10,59.8, 100,178.4, 200,210,220,222,222.5,223,224])

print(coeffs(compose_them(test)))

print(torch.tensor(0.1).float_power(torch.tensor(446.0)))


########## COPILOT ON EVALUATING #############

# import numpy as np
# from make_dataset import read_foods_tensor, FoodProperties as FP
# import torch
# from numpy.polynomial.chebyshev import Chebyshev
# import matplotlib.pyplot as plt

# class RoundSTE(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return torch.round(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output
    
# def round_ste(input):
#     return RoundSTE.apply(input)

# def mask(x):
#     return x * torch.sigmoid(50 * (222.5 - x))

# def compose_them(x):
#     return mask(round_ste(x))

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data = read_foods_tensor()

# x = torch.tensor(range(len(data)), dtype=torch.float32)

# y = data[:, FP.FAT.value]

# # Fit Chebyshev polynomial using numpy
# coeffs = Chebyshev.fit(x.numpy(), y.numpy(), 446)

# # Convert numpy coefficients to torch tensor
# coeffs_torch = torch.tensor(coeffs.coef, dtype=torch.float32)

# def chebyshev_poly(x, coeffs):
#     T = [torch.ones_like(x), x]
#     for n in range(2, len(coeffs)):
#         T.append(2 * x * T[-1] - T[-2])
#     return sum(c * Tn for c, Tn in zip(coeffs, T))

# # Generate y values for the fitted function using torch
# y_fit = chebyshev_poly(x, coeffs_torch)

# # Plot the results
# plt.scatter(x.numpy(), y.numpy(), label="Data points", color="red")
# plt.plot(x.numpy(), y_fit.detach().numpy(), label="Chebyshev Fit", color="blue")
# plt.legend()
# plt.show()

# print(coeffs_torch)

# test = torch.tensor([0, 1, 10, 59.8, 100, 178.4, 200, 210, 220, 222, 222.5, 223, 224], dtype=torch.float32)

# print(chebyshev_poly(compose_them(test), coeffs_torch))

# print(torch.tensor(0.1).float_power(torch.tensor(446.0)))