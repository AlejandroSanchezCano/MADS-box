from src.entities.protein_protein import ProteinProtein
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

## Get contact maps
ppis = [ppi for ppi in ProteinProtein.iterate(end = 4489) if hasattr(ppi, 'contacts')]
interactions = [ppi.interaction for ppi in ppis]
contacts = [ppi.contacts for ppi in ppis]
max_shape = max([contact.matrix.shape[0] for contact in contacts])
resized = np.array([contact.resize(max_shape, max_shape) for contact in contacts])

# Shuffle and split
indices = np.random.permutation(resized.shape[0])
resized = resized[indices]
interactions = np.array(interactions)[indices]
ntrain = int(len(contacts)*0.8)
train_cm, test_cm = resized[:ntrain], resized[ntrain:]
train_label, test_label = interactions[:ntrain], interactions[ntrain:]
train_cm = np.expand_dims(train_cm, axis=1)
test_cm = np.expand_dims(test_cm, axis=1)

train = list(zip(train_cm, train_label))
test = list(zip(test_cm, test_label))
ttrain =torch.utils.data.DataLoader(train, 
                                          batch_size=32, 
                                          shuffle=True, 
                                          num_workers=1)
    
ttest = torch.utils.data.DataLoader(test, 
                                          batch_size=32, 
                                          shuffle=True, 
                                          num_workers=1)

# NNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 51 * 51, 512)  # 408/2/2/2 = 51
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 51 * 51)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in ttrain:
        labels = labels.float().unsqueeze(1)  # Adjust labels to match output shape
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(ttrain)}")

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in ttest:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")