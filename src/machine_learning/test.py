# Imports
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from src.misc.logger import logger
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from src.entities.protein_protein import ProteinProtein
from src.zen.performance import Performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.setLevel(logging.INFO)

## Get contact maps
ppis = [ppi for ppi in ProteinProtein.iterate() if hasattr(ppi, 'contacts')]
#assert len(ppis) == 5041, "There aren't 5041 contact maps, see total ppi length"
labels = torch.tensor([ppi.interaction for ppi in ppis]).float().to(device)
maps = [ppi.contacts for ppi in ppis]
logger.info(f"{len(maps)} contact maps retrieved")

## Process image (resize/pad/crop)
max_shape = max([map.matrix.shape[0] for map in maps])
resized = [map.resize(max_shape, max_shape) for map in tqdm(maps)]
resized = torch.from_numpy(np.array(resized)).float().to(device)
resized = resized.unsqueeze(1)
logger.info(f'Images resized to shape {resized.shape}')

## Process image (pad)
#max_shape = max([map.matrix.shape[0] for map in maps])
#padded = [map.pad(max_shape, max_shape) for map in tqdm(maps)]
#padded = torch.from_numpy(np.array(padded)).float().to(device)
#padded = padded.unsqueeze(1)
#logger.info(f'Images padded to shape {padded.shape}')

# Data augmentation
max_shape = max([map.matrix.shape[0] for map in maps])
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((max_shape, max_shape)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

images = [transform(image) for image in resized]
images = torch.from_numpy(np.array(images)).float().to(device).unsqueeze(1)
logger.info(f'Images transformed to shape {images.shape}')


images = resized
# Load into DataLoaders
train_maps, test_maps, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, shuffle = True, random_state=42)
logger.info(f"Split ==> Train maps: {train_maps.shape}, Test maps: {test_maps.shape}")

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
        self.fc1 = nn.Linear(128 * int(max_shape/2/2/2)**2, 256)  # 408/2/2/2 = 51
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        logger.debug(f'Convolution 1 {x.shape}')
        x = self.pool(torch.relu(self.conv2(x)))
        logger.debug(f'Convolution 2 {x.shape}')
        x = self.pool(torch.relu(self.conv3(x)))
        logger.debug(f'Convolution 3 {x.shape}')
        x = x.view(-1, 128 * int(max_shape/2/2/2)**2)  # Flatten the tensor
        logger.debug(f'Flatten {x.shape}')
        x = torch.relu(self.fc1(x))
        logger.debug(f'FC1 {x.shape}')
        x = self.fc2(x)
        logger.debug(f'FC2 {x.shape}')
        x = self.sigmoid(x)
        logger.debug(f'Sigmoid {x.shape}')
        return x

# Neural network
class SimplerCNN(nn.Module):
    def __init__(self):
        super(SimplerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * int(max_shape/2)**2, 64)  # 408/2/2/2 = 51
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        logger.debug(f'Convolution 1 {x.shape}')
        x = x.view(-1, 32 * int(max_shape/2)**2)  # Flatten the tensor
        logger.debug(f'Flatten {x.shape}')
        x = torch.relu(self.fc1(x))
        logger.debug(f'FC1 {x.shape}')
        x = self.dropout(x)
        logger.debug(f'Dropout {x.shape}')
        x = self.fc2(x)
        logger.debug(f'FC2 {x.shape}')
        x = self.sigmoid(x)
        logger.debug(f'Sigmoid {x.shape}')
        return x

# Train
def train_model(model, criterion, optimizer, loader):
    # Train model
    model.train()
    # Accumulate loss
    total_loss = 0
    # Batch iterator
    for images, labels in loader:
        x = Variable(images)        # Batch x
        y = Variable(labels)        # Batch y
        y = y.float().unsqueeze(1)  # Add dimension
        output = model(x)           # Forward pass
        loss = criterion(output, y) # Compute loss
        optimizer.zero_grad()       # Clear gradients
        loss.backward()             # Backpropagation
        optimizer.step()            # Update weights
        total_loss += loss.item()   # Accumulate loss
    return total_loss / len(loader) # Return average loss

# Test
def evaluate_model(model, criterion, loader):
    # Eval model
    model.eval()
    # Accumulate loss
    total_loss = 0
    # Accumulate labels and predictions
    total_labels = np.array([])
    total_predicted = np.array([])
    # Batch iterator
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            loss = criterion(outputs, labels)
            total_labels = np.append(total_labels, labels.cpu().numpy())
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())
            total_loss += loss.item()

    return total_loss / len(loader), total_labels, total_predicted

# Main
model = CNN().to(device)
#model = SimplerCNN().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
num_epochs = 200

# Train + evaluate
train_losses, test_losses, test_accuracies, train_accuracies = [], [], [], []
for epoch in tqdm(range(num_epochs)):
    # Train
    train_loss = train_model(model, criterion, optimizer, train)
    # Evaluate with train data
    train_loss, train_labels, train_predicted = evaluate_model(model, criterion, train)
    # Evaluate with test data
    test_loss, test_labels, test_predicted = evaluate_model(model, criterion, test)
    # Performance
    train_performance = Performance(train_labels, train_predicted)
    test_performance = Performance(test_labels, test_predicted)
    # Logging
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_performance.balanced_accuracy}, Train Accuracy: {train_performance.balanced_accuracy}')
    # Append
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_performance.balanced_accuracy)
    train_accuracies.append(train_performance.balanced_accuracy)

# Save model    
bacc = np.mean(test_accuracies[-10:])
torch.save(model.state_dict(), f'models/{bacc:.2f}.pth')
print(bacc)

# Plot confussion matrix
test_performance.plot_confusion_matrix()

# Plot losses
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')
plt.savefig('losses.png')
plt.clf()

# Plot accuracies
plt.plot(train_accuracies)
plt.plot(test_accuracies)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Balanced Accuracies')
plt.savefig('accuracies.png')
plt.clf()

