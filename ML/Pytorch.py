import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([16.8, 29.3, 41.8, 54.3], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)
b = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

def forward(x):
    return x*w + b

loss = nn.MSELoss()
opt = torch.optim.SGD([w, b], lr=0.01)

n_iter = 100000

for i in range(n_iter):
    y_pred = forward(X)
    
    l = loss(Y, y_pred) 

    l.backward()

    opt.step()

    opt.zero_grad()

    if(l < 0.00000001):
        print(f'w = {w:.3f}, b = {b:.3f}, loss = {l:.8f}, y_pred = {y_pred}, f10 = {forward(10):.3f}')
        break

