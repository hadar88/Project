from make_dataset import read_foods_tensor, FoodProperties as FP
import torch
from numpy.polynomial.chebyshev import Chebyshev

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

def chebyshev_eval(x, coefficients):
    """ Evaluate a Chebyshev polynomial expansion using PyTorch. """
    # Normalize x to [-1, 1] range for Chebyshev evaluation
    x_norm = x / 111 - 1
    
    # Initialize Chebyshev polynomials
    T0 = torch.ones_like(x_norm)
    T1 = x_norm
    
    # Compute the polynomial expansion
    result = coefficients[0] * T0 + coefficients[1] * T1
    
    for n in range(2, len(coefficients)):
        T2 = 2 * x_norm * T1 - T0
        result += coefficients[n] * T2
        T0, T1 = T1, T2
    
    return result

print("Reading Data...")
# Load data
data = read_foods_tensor()

# Prepare x and y
x = torch.tensor(range(len(data)))
y = data[:, FP.VEGETARIAN.value]

def multiple_bell_curves(x):
    global y

    return torch.sum(torch.stack([v * torch.exp(-((x - i * v) ** 2) / 0.00001) for i, v in enumerate(y)], dim=0), dim=0)

print("Fitting...")
# Fit Chebyshev polynomial using NumPy
# coeffs = Chebyshev.fit(x, y, 446)

# Convert NumPy coefficients to list
# coefficients = torch.tensor(coeffs.coef)

# # Generate y values using the PyTorch Chebyshev evaluation
# y_fit = chebyshev_eval(x, coefficients)

# # Optionally plot the results
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, label="Original Data", color="red", alpha=0.5)
# plt.plot(x, y_fit, label="Chebyshev Fit", color="blue")
# plt.title("Chebyshev Polynomial Fit")
# plt.xlabel("Index")
# plt.ylabel("Fat Content")
# plt.legend()
# plt.show()

# Test with different input values
# test = torch.tensor([0, 1, 10, 59.8, 100, 178.4, 200, 210, 220, 222, 222.5, 223, 224])
test = torch.tensor([
6, 12, 237.45, 0.83, 189, 76.29, 143, 251.67, 45, 112.56, 203,
18.72, 134, 89.34, 211, 5.67, 167, 42.91, 226, 98, 155.23,
67.45, 11, 242.78, 33, 176.54, 109, 54.12, 201.36, 87, 22.65,
139, 71.89, 216, 39.47, 184, 62.13, 247, 105, 19.76, 230,
53, 127.42, 94.56, 172, 7.31, 241, 115, 66.28, 198, 36.95, 0.00
])


print("Evaluating...")
# test_fit = chebyshev_eval(compose_them(test), coefficients)

# printable = [(test[i], test_fit[i]) for i in range(len(test))]

print(compose_them(torch.tensor(241.00)))

for x in test:
    print(f"{x:.2f} -> {multiple_bell_curves(compose_them(x)):.2f}")

# print("Test input:", test)
# # print the results line by line
# for i in range(len(printable)):
#     print(f"Input: {printable[i][0]:.2f}, Fit: {printable[i][1]:.2f}")
