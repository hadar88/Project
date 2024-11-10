import torch
import torch.nn as nn
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# (x, y) -> (10x + 5y + 1, 2.5x + -9.5y + 8)
X = torch.tensor([[1, 2], [2, 3], [3,5], [4, 8]], dtype = torch.float32)
Y = torch.tensor([[21, -8.5], [36, -15.5], [56, 32], [81, -58]], dtype = torch.float32)

test = torch.tensor([10, 9], dtype = torch.float32) # (146, -52.5)

n_features = X.shape[1]

input_size = n_features
output_size = Y.shape[1]

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)


loss = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

n_iter = 1000000

for i in range(n_iter):
    y_pred = model(X)
    
    l = loss(Y, y_pred) 

    l.backward()

    opt.step()

    opt.zero_grad()

    if(l < 0.00001):
        for name, param in model.named_parameters():
            print(f"Parameter value: {param.data}")
        print(f'loss = {l:.8f}, y_pred = {y_pred}, f10 = {model(test)}')
        break


"""
predicted = model(X).detach()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0].tolist(), X[:,1].tolist(), Y.squeeze().tolist(), c='r', marker='o')
ax.scatter(X[:, 0].tolist(), X[:,1].tolist(), predicted, c='b', marker='^')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')

plt.show()"""