import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import math
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import time

# Setting device to cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Global variables
epochs = 10
lr = 1e-3
batch_size = 10
input_size = 100
num_samples = 100

random_samples = sorted([random.uniform(0, 4*math.pi) for x in range(total_indeces)])

input = torch.tensor([math.sin(number) for number in random_samples])
input = input.to(device)

print(input.is_cuda)


# Defining our neural net class
class Net(nn.Module):
    
    def __init__(self):

        # initialising the nn.Module
        super(Net, self).__init__()

        # input layer followed by two hidden layers each of size 512,
        # with relu activations after each hidden layer

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    # Definining the forward pass, which simply pushes the input through the layers
    def forward(self, input):
        out = self.linear_relu_stack(input)
        return out

# Initialises an instance of our neural net and ensuring it's on GPU
net = Net().cuda()
print(net)

# gets the number of matrices of weights and biases,
# 6 in this case (3x weights, 3x biases) for the three layers
params = list(net.parameters())
print(len(params))
#extracts the dimensions of the first hidden layer's weights matrix
print(params[0].size())

#setting loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1)

#learning
for i in range(100):
    optimizer.zero_grad()     #zeros gradient buffers
    output = net(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    print(loss)

    #drawing the learnt curve at each iteration
    drawinput = (input.cpu()).numpy()
    drawoutput = (output.cpu()).detach().numpy()

    plt.plot(random_samples, drawinput, label = 'input', color = 'blue', linestyle = "-")

    plt.plot(random_samples, drawoutput, label = 'output', color = 'red', linestyle = "-")
    plt.grid(True)
   #plt.show()
    #time.sleep(0.5)


print(loss)

drawinput = (input.cpu()).numpy()
drawoutput = (output.cpu()).detach().numpy()
print(drawinput)
print(drawoutput)

plt.plot(random_samples, drawinput, label = 'input', color = 'blue', linestyle = "-")

plt.plot(random_samples, drawoutput, label = 'output', color = 'red', linestyle = "-")
plt.grid(True)
plt.show()

