
# Third-party tools
import torch
import torch.nn as nn

class CCCCLL(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CCCCLL, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * int(max_shape/2/2/2/2)**2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = self.pool(torch.relu(self.conv2(x)))            # Convolution 2
        x = self.pool(torch.relu(self.conv3(x)))            # Convolution 3
        x = self.pool(torch.relu(self.conv4(x)))            # Convolution 4
        x = x.view(-1, 256 * int(self.max_shape/2/2/2/2)**2)  # Flatten
        x = torch.relu(self.fc1(x))                         # Fully connected 1
        x = self.dropout(x)                                 # Dropout
        x = self.fc2(x)                                     # Fully connected 2
        x = self.sigmoid(x)                                 # Sigmoid
        return x
    
class CNN(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CNN, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * int(max_shape/2/2/2)**2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = self.pool(torch.relu(self.conv2(x)))            # Convolution 2
        x = self.pool(torch.relu(self.conv3(x)))            # Convolution 3
        x = x.view(-1, 128 * int(self.max_shape/2/2/2)**2)  # Flattem
        x = torch.relu(self.fc1(x))                         # Fully connected 1
        x = self.dropout(x)                                 # Dropout
        x = self.fc2(x)                                     # Fully connected 2
        x = self.sigmoid(x)                                 # Sigmoid
        return x
    

class CCCL(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CCCL, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * int(max_shape/2/2/2)**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = self.pool(torch.relu(self.conv2(x)))            # Convolution 2
        x = self.pool(torch.relu(self.conv3(x)))            # Convolution 3
        x = x.view(-1, 128 * int(self.max_shape/2/2/2)**2)  # Flatten
        x = self.fc1(x)                                     # Fully connected 1
        x = self.sigmoid(x)                                 # Sigmoid
        return x

class CCL(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CCL, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * int(max_shape/2/2)**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = self.pool(torch.relu(self.conv2(x)))            # Convolution 2
        x = x.view(-1, 64 * int(self.max_shape/2/2)**2)     # Flatten
        x = self.fc1(x)                                     # Fully connected 1
        x = self.sigmoid(x)                                 # Sigmoid
        return x

class CL(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CL, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * int(max_shape/2)**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = x.view(-1, 32 * int(self.max_shape/2)**2)       # Flatten
        x = self.fc1(x)                                     # Fully connected 1
        x = self.sigmoid(x)                                 # Sigmoid
        return x

class CNN2Channels(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CNN2Channels, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * int(max_shape/2/2/2)**2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))            # Convolution 1
        x = self.pool(torch.relu(self.conv2(x)))            # Convolution 2
        x = self.pool(torch.relu(self.conv3(x)))            # Convolution 3
        x = x.view(-1, 128 * int(self.max_shape/2/2/2)**2)  # Flattem
        x = torch.relu(self.fc1(x))                         # Fully connected 1
        x = self.dropout(x)                                 # Dropout
        x = self.fc2(x)                                     # Fully connected 2
        x = self.sigmoid(x)                                 # Sigmoid
        return x
    