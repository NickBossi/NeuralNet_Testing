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
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np

from sklearn.model_selection import KFold

# Setting the device to cuda 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Hypeparameters
batch_size = 4
learning_rate = 0.01
l2 = 0.01
l2_list = np.linspace(0,0.5,20)

# Dimensions of data
input_size = 100
num_samples = 400

numbers = sorted([random.uniform(0,4*math.pi) for x in range(input_size)])
random_sample = torch.tensor([math.sin(number) for  number in numbers])
# print(random_sample.shape)

#random_samples = sorted([random.uniform(0, 4*math.pi) for x in range(input_size)])
numbers1 = torch.tensor([sorted([random.uniform(0, 4*math.pi) for x in range(input_size)]) for y in range(num_samples)])
#print(numbers1[:,0])

input = numbers1
response = torch.sin(input)

class CustomDataset(Dataset):
    def __init__(self, input_tensor, output_tensor, transform = None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.mean = self.input_tensor.mean(dim = 0, keepdim = True)
        self.std = self.input_tensor.std(dim = 0, keepdim = True)
        self.transform = transform

    def __len__(self):
        return(len(self.input_tensor))
    
    def __getitem__(self, index):

        x = self.input_tensor[index]
        y = self.output_tensor[index]

        # Normalizing the input data
        x = (x-self.mean)/self.std
        #if self.transform:
        #   x = self.transform(x)
        
        return x,y
    

# Normalization transformation
normalize = transforms.Normalize(torch.mean(input), torch.std(input))

data = CustomDataset(input, response)
dataload = DataLoader(data, batch_size=batch_size)

# # Testing dataloader
# for batch_idx, batch in enumerate(dataload):
#     print(batch_idx)
#     inputs, targets = batch
#     print("Input tensor: ")
#     print(inputs.shape)
#     print("Output tensor: ")
#     print(targets.shape)
    
# Defining our neural net
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

    # Defining the forward pass, which simply pushes the input through the layers
    def forward(self, input):
        out = self.linear_relu_stack(input)
        return out

loss_fn = nn.MSELoss()

# Defining out training function
def train(model, device, train_loader, optimizer, epoch):
    model.train(True)
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(inputs).squeeze(1)

        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss              #accumulates total loss
    train_loss = train_loss/(i+1)       #gets average loss over batches
    return train_loss                   #returns average loss

######################################## K fold cross validation ##################################################

#sets number of epoch and best_v_loss which will be used to track which model gives best validation performance
EPOCHS = 8
best_v_loss = 1000000
epoch_number = 0

k_folds = 9

kf = KFold(n_splits=k_folds, shuffle = True)

#Setting model and optimizer
model = Net().to(device)

for i in l2_list:

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=i)

    for fold, (train_idx,test_idx) in enumerate(kf.split(data)):
        print("++++++++++++++++++")
        print("Fold {}".format(fold+1))
        print("++++++++++++++++++")

        #Defining dataloaders for current fold

        train_load = DataLoader(dataset=data,
                                batch_size=batch_size,
                                sampler = torch.utils.data.SubsetRandomSampler(test_idx))
        
        valid_load = DataLoader(dataset= data, 
                                batch_size = batch_size,
                                sampler = torch.utils.data.SubsetRandomSampler(test_idx))
        
        # Training all epoch for each fold

        for epoch in range(EPOCHS):
            print('EPOCH {}'.format(epoch+1))
            print("-------------------")
            
            avg_t_loss = train(model, device, train_load, optimizer, epoch)   #trains for an epoch and then outputs average loss over epoch

            #Turns on model evaluation, more efficient for validation as not training
            running_vloss = 0.0
            model.eval()

            # Disables gradient tracking, reduces memory consumption
            with torch.no_grad():
                for i, vdata in enumerate(valid_load):
                    vinputs, vtargets = vdata
                    vinputs = vinputs.to(device)
                    vtargets = vtargets.to(device)
                    vouptuts = model(vinputs).squeeze(1)
                    vloss = loss_fn(vouptuts, vtargets)
                    running_vloss +=vloss
            
            avg_v_loss = running_vloss/(i+1)            #calculates average validation loss
            print('LOSS train {} valid {}'.format(avg_t_loss, avg_v_loss))            #outputs training vs validation loss

            # Checks to see if validation loss has improved and saves the model as the optimal model
            if avg_v_loss < best_v_loss:
                best_v_loss = avg_v_loss
                optimal_model = copy.deepcopy(model)
                #model_path = 'model_{}'.format(epoch_number+1)
                #torch.save(model.state_dict(), model_path)

pred_input = torch.tensor(numbers)
pred_input = (pred_input - torch.mean(pred_input))/torch.std(pred_input)
prediction = optimal_model(torch.tensor(pred_input).to(device))

plt.plot(numbers, random_sample, label = 'input', color = 'blue', linestyle = "-")

plt.plot(numbers, prediction.cpu().detach().numpy(), label = 'output', color = 'red', linestyle = "-")
plt.grid(True)
plt.show()

