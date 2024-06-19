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
from src.machine_learning.performance import Performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.setLevel(logging.INFO)

## Get contact maps
ppis = [ppi for ppi in ProteinProtein.iterate() if hasattr(ppi, 'contacts')][:10]
#assert len(ppis) == 5041, "There aren't 5041 contact maps, see total ppi length"
labels = torch.tensor([ppi.interaction for ppi in ppis]).float().to(device)
maps = [ppi.contacts for ppi in ppis]
logger.info(f"{len(maps)} contact maps retrieved")

# Process image (resize/pad/crop)
max_shape = max([map.matrix.shape[0] for map in maps])
resized = [map.resize(max_shape, max_shape) for map in tqdm(maps)]
resized = torch.from_numpy(np.array(resized)).float().to(device)
resized = resized.unsqueeze(1)
logger.info(f'Images resized to shape {resized.shape}')

# Process image (pad)
# max_shape = max([map.matrix.shape[0] for map in maps])
# padded = [map.pad(max_shape, max_shape) for map in tqdm(maps)]
# padded = torch.from_numpy(np.array(padded)).float().to(device)
# padded = padded.unsqueeze(1)
# logger.info(f'Images padded to shape {padded.shape}')

processed = resized

# Data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(50, 0, padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.ToTensor()
])

image = processed[0]
print(image.shape)
image = image.permute(1, 2, 0)
print(image.shape)
image = image.cpu().numpy()
t = transform(image)
y = transform(image)
print(t.shape)
print(t[:,:,0].shape)
print(torch.equal(t,y))

#plot
plt.imshow(t[0,:,:], cmap = "Greys")
plt.savefig('t')
plt.imshow(y[0,:,:], cmap = "Greys")
plt.savefig('y')