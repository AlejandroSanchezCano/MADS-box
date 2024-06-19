# Imports
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from src.misc.logger import logger
from sklearn.model_selection import train_test_split
from src.entities.protein_protein import ProteinProtein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.setLevel(logging.INFO)

## Get contact maps
ppis = [ppi for ppi in ProteinProtein.iterate_paper_mikc() if hasattr(ppi, 'contacts')]
#assert len(ppis) == 5041, "There aren't 5041 contact maps, see total ppi length"
labels = torch.tensor([ppi.interaction for ppi in ppis]).float().to(device)
maps = [ppi.contacts.ab(ppi.p1, ppi.p2) for ppi in ppis]
logger.info(f"{len(maps)} contact maps retrieved")

# Process image (resize/pad/crop)
#max_shape = max([map.matrix.shape[0] for map in maps])
#resized = [map.resize(max_shape, max_shape) for map in tqdm(maps)]
#resized = torch.from_numpy(np.array(resized)).float().to(device)
#resized = resized.unsqueeze(1)
#logger.info(f'Images resized to shape {resized.shape}')

# Process image (pad)
max_shape = max([map.matrix.shape[0] for map in maps])
padded = [map.pad(max_shape, max_shape) for map in tqdm(maps)]
padded = torch.from_numpy(np.array(padded)).float().to(device)
padded = padded.unsqueeze(1)
logger.info(f'Images padded to shape {padded.shape}')
#padded = padded[:,:,50:100, 50:100]


processed = padded

# Load into DataLoaders
train_maps, test_maps, train_labels, test_labels = train_test_split(processed, labels, test_size=0.2, shuffle = True, random_state=42)
train = torch.utils.data.TensorDataset(train_maps, train_labels)
test = torch.utils.data.TensorDataset(test_maps, test_labels)
train = torch.utils.data.DataLoader(train, batch_size = 100, shuffle=True)
test = torch.utils.data.DataLoader(test, batch_size = 100, shuffle=False)

# Neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1 * int(processed.shape[-1])**2, 64)  # 408/2/2/2 = 51
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.pool(torch.relu(self.conv1(x)))
        #logger.debug(f'Convolution 1 {x.shape}')
        #x = self.pool(torch.relu(self.conv2(x)))
        #logger.debug(f'Convolution 2 {x.shape}')
        #x = self.pool(torch.relu(self.conv3(x)))
        #logger.debug(f'Convolution 3 {x.shape}')
        logger.debug(f'Input {x.shape}')
        x = x.view(-1, 1 * int(processed.shape[-1])**2)  # Flatten the tensor
        logger.debug(f'Flatten {x.shape}')
        x = torch.relu(self.fc1(x))
        logger.debug(f'FC1 {x.shape}')
        #x = self.dropout(x)
        #logger.debug(f'Dropout {x.shape}')
        x = self.fc2(x)
        x = torch.relu(x)
        logger.debug(f'FC2 {x.shape}')
        x = self.fc3(x)
        x = self.sigmoid(x)
        logger.debug(f'Sigmoid {x.shape}')
        return x
    
# Train
cnn = CNN().to(device)
loss_func = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.0001) 
cnn.train()
num_epochs = 1000
total_step = len(train)
train_loss = []
for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for i, (images, labels) in enumerate(train):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            b_y = b_y.float().unsqueeze(1)
            output = cnn(b_x)          
            loss = loss_func(output, b_y)
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            #
            running_loss += loss.item()
        train_loss.append(running_loss / len(train))

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train)}')

# Test
test_loss = []
for epoch in tqdm(range(num_epochs)):
    cnn.eval()
    running_loss = 0
    with torch.no_grad():
        total_labels = np.array([])
        total_predicted = np.array([])
        for images, labels in test:
            labels = labels.float().unsqueeze(1)
            outputs = cnn(images)
            predicted = (outputs > 0.5).float()
            loss = loss_func(outputs, labels)

            total_labels = np.append(total_labels, labels.cpu().numpy())
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())

            running_loss += loss.item()
        test_loss.append(running_loss / len(train))

from src.machine_learning.performance import Performance
performance = Performance(total_labels, total_predicted)
print(total_labels.shape, total_predicted.shape)
print(performance.balanced_accuracy)
print(performance.f1)
print(performance.mcc)
print(performance.confusion_matrix)
performance.plot_confusion_matrix()
performance.plot_roc_curve()

import matplotlib.pyplot as plt
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('losses')
plt.clf()