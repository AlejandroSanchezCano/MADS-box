import torch
import torch.nn as nn
from torchvision import datasets
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from src.misc.logger import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download data
train = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
test = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
logger.info(train)
logger.info(test)

# Load data
train = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, num_workers=1)
test = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True, num_workers=1)

# Neural network
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # CNN layer 1
        self.conv1 =  nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=5, stride=1, padding=2)                  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = None, padding = 0)
        # CNN layer 2
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=5, stride=1, padding=2)                 
        # Fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    # TODO: sigmoid softmax
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # conv1 -> relu -> pool
        x = self.pool(self.relu(self.conv2(x))) # conv2 -> relu -> pool
        x = x.view(x.size(0), -1) #flatten (batch_size, 32 * 7 * 7)       
        # print(x.view(x.size(0), -1))
        # print(x.view(x.size(0), -1).shape) # 100, 1568 (32 * 7 * 7)
        # print(x.view(-1, x.size(0)))
        # print(x.view(-1, x.size(0)).shape) # 1568 (32 * 7 * 7), 100

        x = self.fc1(x) # fully connected layer

        return self.fc2(x)

# Train
cnn = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01) 

cnn.train()
num_epochs = 10
total_step = len(train)
for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)           
            print(output, b_y)  
            loss = loss_func(output, b_y)
        
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                
# Test
cnn.eval()
with torch.no_grad():
    accuracies = []
    for images, labels in test:
        test_output = cnn(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()

        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print(f'Test Accuracy of the model on the 10000 test images: {accuracy}')
        accuracies.append(accuracy)

print(f'Average accuracy: {sum(accuracies) / len(accuracies)}')
    