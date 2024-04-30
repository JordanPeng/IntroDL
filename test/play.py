import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # Example tensor initialization
#         self.A = torch.randn(10, 10)
#         self.B = torch.randn(10, 10)
#         self.C= torch.randn(10, 10)
#         self.D = torch.randn(10, 10)
#
#         # Ensure A and B are Parameters if they're meant to be trainable
#         self.A = nn.Parameter(self.A)
#         self.B = nn.Parameter(self.B)
#         self.C = nn.Parameter(self.C)
#         self.D = nn.Parameter(self.D)
#
#
#         self.layer1 = nn.Linear(10, 10)
#         self.layer2 = nn.Linear(10, 10)
#     def forward(self, x):
#         print(model.layer1.weight)
#         self.layer1.weight = nn.Parameter(self.layer1.weight + self.A @ self.B)
#         print(model.layer1.weight)
#         self.layer2.weight = nn.Parameter(self.layer2.weight + self.C @ self.D)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#
#         return x
#
# model = MyModel()
# model.to(device)
# model.train()
# for name, param in model.named_parameters():
#     print(name)
#
# x= (torch.randn(10, 10)*0.01).to(device)
# y= (torch.randn(10, 10)*0.01).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer.zero_grad()
# output = model(x)
# loss = torch.nn.functional.mse_loss(output, y)
# loss.backward()
# optimizer.step()
#
# print(model.layer1.weight)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initializing matrices A, B, C, D as Parameters if they need to be trainable
        self.register_parameter("A", nn.Parameter(torch.randn(10, 10)))
        self.register_parameter("B", nn.Parameter(torch.randn(10, 10)))
        self.register_parameter("C", nn.Parameter(torch.randn(10, 10)))
        self.register_parameter("D", nn.Parameter(torch.randn(10, 10)))

        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        # Apply modifications using operations that maintain gradient tracking
        modified_weight1 = self.layer1.weight + torch.matmul(self.A, self.B)
        modified_weight2 = self.layer2.weight + torch.matmul(self.C, self.D)

        # Use functional interface to apply weights explicitly
        x = nn.functional.linear(x, modified_weight1, self.layer1.bias)
        x = nn.functional.linear(x, modified_weight2, self.layer2.bias)
        return x

model = MyModel()
model.to(device)
model.train()

# Print parameter names
for name, param in model.named_parameters():
    print(name)

# Sample input and target
x = (torch.randn(10, 10) * 0.01).to(device)
y = (torch.randn(10, 10) * 0.01).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()

# Forward pass, loss computation, and backward pass
output = model(x)
loss = torch.nn.functional.mse_loss(output, y)
loss.backward()
optimizer.step()

# Check updated weight
print(model.layer1.weight)