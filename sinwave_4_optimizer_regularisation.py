import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader

# Global Parameters
regularisations = np.linspace(0,1,20)

# Setting up the Dataset object
class SinDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = torch.tensor(data, dtype = torch.float32)
        self.transform = transform 


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        #applies transformation when item is called
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Setting the device to cuda 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Dimensions of data
input_size = 500
num_samples = 20


numbers = sorted([random.uniform(0,4*math.pi) for x in range(input_size)])
random_sample = torch.tensor([math.sin(number) for  number in numbers]).to(device) 
# print(random_sample.shape)

#random_samples = sorted([random.uniform(0, 4*math.pi) for x in range(input_size)])
input_data = torch.tensor([([math.sin(number) for number in sorted([random.uniform(0, 4*math.pi) for x in range(input_size)])]) for i in range(num_samples)])

# Splitting into Training, Testing and Validation Sets
train_data = input_data[:16].to(device)
validation_data = input_data[16:19].to(device)
test_data = input_data[19:].to(device)



# Normalization transformation
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.mean(input_data), torch.std(input_data))])

# Creating datasets for each
train_set = SinDataset(train_data, transform=normalize)
validation_set = SinDataset(validation_data, transform = normalize)
test_set = SinDataset(test_data, transform = normalize)

# Creating DataLoaders for each
train_loader = DataLoader(train_data,  batch_size = 4, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size = 2, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 2, shuffle = True)

print('Training set has {} instances'.format(len(train_set)))
print('Validation set has {} instances'.format(len(validation_set)))

#input = torch.tensor([math.sin(number) for number in random_samples])
input_data = input_data.to(device)

#print(input_data[:,0])

dataset = SinDataset(input_data, transform=normalize)

dataloader = DataLoader(dataset, batch_size=10, shuffle = True)


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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    # Definining the forward pass, which simply pushes the input through the layers
    def forward(self, input):
        out = self.linear_relu_stack(input)
        return out

# Initialises an instance of our neural net and ensuring it's on GPU
net = Net().cuda()
#print(net)

# gets the number of matrices of weights and biases,
# 6 in this case (3x weights, 3x biases) for the three layers
params = list(net.parameters())

#print(len(params))
#extracts the dimensions of the first hidden layer's weights matrix
#print(params[0].size())

#setting loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01, weight_decay=0.05)


# Defining the training over an epoch
def train_one_epoch(epoch_index):
    for i, data in enumerate(train_loader):

        running_loss = 0 
        last_loss = 0

        input = data

        optimizer.zero_grad()

        output = net(input)

        loss = criterion(output, input)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 4 == 3:                 #batch size is 4
            last_loss = running_loss/4 #loss per batch
            print(' batch {} loss {}'.format(i+1, last_loss))
            running_loss = 0
        print('epoch {}: loss {}'.format(epoch_index+1, last_loss))
    
    # Gives loss of training
    return last_loss

#timestamp = time.datetime.now().strftime('%Y%m%d_%H%M%S')

EPOCHS = 50
best_v_loss = 1000000
epoch_number = 0


# Validation over epochs
for epoch in range(EPOCHS):
    print('EPOCH {}'.format(epoch_number+1))

    #ensures gradient tracking is on
    net.train(True)
    avg_loss = train_one_epoch(epoch_number)   #gets loss associated with training epoch

    #Turns on model evaluation, more efficient for validation as not training
    running_vloss = 0.0
    net.eval()

    # Disables gradient tracking, reduces memory consumption
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vouptuts = net(vdata)
            vloss = criterion(vouptuts, vdata)
            running_vloss +=vloss
    
    avg_vloss = running_vloss/(i+1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    if avg_vloss < best_v_loss:
        best_v_loss = avg_vloss
        model_path = 'model_{}'.format(epoch_number+1)
        torch.save(net.state_dict(), model_path)
    
    epoch_number +=1


#Recovering optimal model 
saved_model = Net()
print(model_path)
saved_model.load_state_dict(torch.load(model_path))

#Making final prediction and plotting
test_data = test_data.cpu()   #puts it back on cpu for prediction and plotting
final_pred = saved_model(test_data)

#
sindata = torch.tensor([math.sin(number) for  number in test_data.flatten()])
drawinput = (test_data).numpy()
drawoutput = (final_pred.cpu()).detach().numpy()
print(drawinput)
print(drawoutput)

#plt.plot(test_data, drawinput, label = 'input', color = 'blue', linestyle = "-")

plt.plot(test_data, drawoutput, label = 'output', color = 'red', linestyle = "-")
plt.grid(True)
plt.show()


# #learning
# for i in range(100):
#     optimizer.zero_grad()     #zeros gradient buffers
#     output = net(input)
#     loss = criterion(output, input)
#     loss.backward()
#     optimizer.step()
#     print(loss)

#     #drawing the learnt curve at each iteration
#     drawinput = (input.cpu()).numpy()
#     drawoutput = (output.cpu()).detach().numpy()

#     plt.plot(random_samples, drawinput, label = 'input', color = 'blue', linestyle = "-")

#     plt.plot(random_samples, drawoutput, label = 'output', color = 'red', linestyle = "-")
#     plt.grid(True)
#    #plt.show()
#     #time.sleep(0.5)

'''
print(loss)

drawinput = (input.cpu()).numpy()
drawoutput = (output.cpu()).detach().numpy()
print(drawinput)
print(drawoutput)

plt.plot(random_samples, drawinput, label = 'input', color = 'blue', linestyle = "-")

plt.plot(random_samples, drawoutput, label = 'output', color = 'red', linestyle = "-")
plt.grid(True)
plt.show()
'''


'''
# Getting mean and std for normalisation (not sure this can actually be used)
def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    for data in loader:
        mean += torch.mean(data)
        std += torch.std(data)
    mean = mean/len(loader)
    std = std/len(loader)
    return mean, std
'''