import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([7, 12, 17, 22], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)
b = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

def forward(x):
    return x*w + b

loss = nn.MSELoss()
opt = torch.optim.SGD([w, b], lr=0.01)

n_iter = 1000

for i in range(n_iter):
    y_pred = forward(X)
    
    l = loss(Y, y_pred) 

    l.backward()

    opt.step()

    opt.zero_grad()

    if(i == 999):
        print(f'w = {w}, b = {b}, loss = {l}, y_pred = {y_pred}, dw = {w.grad}, f10 = {forward(10)}')

