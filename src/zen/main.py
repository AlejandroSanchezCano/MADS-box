
# Third-party modules
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from src.entities.protein_protein import ProteinProtein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom modules
from src.zen import cnn
from src.misc.logger import logger
from src.zen.performance import Performance

# Download contact maps
ppis = [ppi for ppi in ProteinProtein.iterate() if hasattr(ppi, 'contacts')]
labels = torch.tensor([ppi.interaction for ppi in ppis]).float().to(device)
maps = [ppi.contacts.i(ppi.p1, ppi.p2) for ppi in ppis]
logger.info(f"{len(maps)} contact maps retrieved")

# Process contact maps
max_shape = max([map.matrix.shape[0] for map in maps])
images = [map.pad(max_shape, max_shape) for map in tqdm(maps)]
images = torch.from_numpy(np.array(images)).float().to(device)
images = images.unsqueeze(1)
logger.info(f'Images resized to shape {images.shape}')

# Split
train_maps, test_maps, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2, shuffle = True, random_state = 42)
logger.info(f"Split ==> Train maps: {train_maps.shape}, Test maps: {test_maps.shape}")
logger.info(f"Split ==> Train labels: {train_labels.shape}, Test labels: {test_labels.shape}")

# Load into DataLoaders
train = torch.utils.data.TensorDataset(train_maps, train_labels)
test = torch.utils.data.TensorDataset(test_maps, test_labels)
train = torch.utils.data.DataLoader(train, batch_size = 100, shuffle=True)
test = torch.utils.data.DataLoader(test, batch_size = 100, shuffle=False)

# Train
def train_model(model, criterion, optimizer, loader):
    # Train model
    model.train()
    # Accumulate loss
    total_loss = 0
    # Batch iterator
    for x, y in loader:
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
model = cnn.CNN(max_shape).to(device)
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
print('aa')

# Plot confussion matrix
test_performance.plot_confusion_matrix()

# Plot losses
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')
if max(train_losses) > 1:
    plt.ylim(0, 1)
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

