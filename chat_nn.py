import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input size: 3 (x, y, t), Output size: 64
        self.fc2 = nn.Linear(64, 64)  # Hidden layer 1: Output size: 64
        self.fc3 = nn.Linear(64, 64)  # Hidden layer 2: Output size: 64
        self.fc4 = nn.Linear(64, 2)   # Output layer: Output size: 2 (f_x, f_y)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Prepare the data
t = np.linspace(0, 1, 100)
x = np.cos(t)
y = np.sin(t)

input_data = torch.tensor(np.column_stack((x, y, t)), dtype=torch.float32)
output_data = torch.tensor(np.column_stack((x**2 + y - t, x * y + t)), dtype=torch.float32)

# Define the neural network, loss function, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train the neural network
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(input_data)
    loss = criterion(outputs, output_data)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))

#Test Model

input = torch.tensor([[0.5,0.5,0.5]], dtype = torch.float32)

# Evaluate the model
with torch.no_grad():
    test_input = input  # Example input
    predicted_output = net(test_input)
    print("Predicted output:", predicted_output.numpy())

#Actual response:
true_response = [input[0][0]**2 + input[0][1] - input[0][2], input[0][0] * input[0][1] + input[0][2]]
print("True ouput: ", true_response)

