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

from sklearn.model_selection import KFold

# Global Parameters
batch_size = 4

# Dimensions of data
input_size = 100
num_samples = 400


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


numbers = sorted([random.uniform(0,4*math.pi) for x in range(input_size)])
random_sample = torch.tensor([math.sin(number) for  number in numbers]).to(device) 
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
        #self.transform = transform

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
    

#plt.plot(input[0,:].detach().numpy(), response[0,:].detach().numpy())
#plt.show()

# Normalization transformation
normalize = transforms.Normalize(torch.mean(input), torch.std(input))

data = CustomDataset(input, response)
dataload = DataLoader(data, batch_size=batch_size)

# Testing dataloader
for batch_idx, batch in enumerate(dataload):
    print(batch_idx)
    inputs, targets = batch
    print("Input tensor: ")
    print(inputs)
    print("Output tensor: ")
    print(targets)
    
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

net = Net().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1)

def train_one_epoch(epoch_index):
    running_loss = 0 
    last_loss = 0
    for i, data in enumerate(dataload):

        input, response = data
        input = input.to(device)
        response = response.to(device)

        optimizer.zero_grad()

        output = net(input).squeeze(1)

        loss = criterion(output, response)
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


for epoch in range(10):
    print('EPOCH {}'.format(epoch+1))

    #ensures gradient tracking is on
    net.train(True)
    avg_loss = train_one_epoch(epoch)   #gets loss associated with training epoch

'''
pred_input=input[0,:].to(device)
pred_input = (pred_input- torch.mean(pred_input))/torch.std(pred_input)
predictions = net(pred_input).cpu()
pred_input=pred_input.cpu().detach().numpy()
plt.plot(pred_input, response[0,:].detach().numpy(), label = "actual", color = "blue", linestyle="-")
plt.plot(pred_input, predictions.detach().numpy(), label = "predicted", color = "red", linestyle="-")
plt.show()
'''

'''
# Loss Function
loss_fn = nn.MSELoss()

# Defining our training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

# K fold cross validation

EPOCHS = 10
best_v_loss = 1000000
epoch_number = 0

k_folds = 9

kf = KFold(n_splits=k_folds, shuffle = True)

print(kf.split(data))

for fold, (train_idx,test_idx) in enumerate(kf.split(data)):
    print("Fold {}".format(fold))
    print("-------------------")

    #Defining dataloaders for current fold

    train_load = DataLoader(dataset=data,
                            batch_size=batch_size,
                            sampler = torch.utils.data.SubsetRandomSampler(test_idx))
    
    valid_load = DataLoader(dataset= data, 
                            batch_size = batch_size,
                            sampler = torch.utils.data.SubsetRandomSampler(test_idx))
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay=0.01)

    for epoch in range(EPOCHS):
        train(model, device, dataload, optimizer, epoch)

'''
'''
input_data = torch.sin(numbers1)
print(input_data.shape)
print(input_data[:,0])
#input_data = torch.tensor([([math.sin(number) for number in sorted([random.uniform(0, 4*math.pi) for x in range(input_size)])]) for i in range(num_samples)])

# Splitting into Training, Testing and Validation Sets
train_data = input_data[:18].to(device)
#validation_data = input_data[16:18].to(device)
test_data = input_data[18:].to(device)



# Creating datasets for each
train_set = SinDataset(train_data, transform=normalize)
#validation_set = SinDataset(validation_data, transform = normalize)
test_set = SinDataset(test_data, transform = normalize)

# Creating DataLoaders for each
train_loader = DataLoader(train_data,  batch_size = 4, shuffle = True)
#validation_loader = DataLoader(validation_data, batch_size = 2, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 2, shuffle = True)

print('Training set has {} instances'.format(len(train_set)))
#print('Validation set has {} instances'.format(len(validation_set)))

#input = torch.tensor([math.sin(number) for number in random_samples])
input_data = input_data.to(device)

#print(input_data[:,0])

dataset = SinDataset(input_data, transform=normalize)

dataloader = DataLoader(dataset, batch_size=10, shuffle = True)



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
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1)
'''





'''
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
'''

'''


        

'''



'''

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
final_pred = saved_model(random_sample.cpu())

drawinput = (torch.sin(numbers1[19,:])).numpy()
drawoutput = (final_pred.cpu()).detach().numpy()
print(drawinput)
print(drawoutput)


plt.plot(numbers1[19,:].numpy(), drawinput, label = 'input', color = 'blue', linestyle = "-")

plt.plot(numbers1[19,:].numpy(), drawoutput, label = 'output', color = 'red', linestyle = "-")
plt.grid(True)
plt.show()

'''