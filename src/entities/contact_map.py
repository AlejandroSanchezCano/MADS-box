
# Third-party modules
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class ContactMap():

    def __init__(self, matrix: np.array):
        self.matrix = matrix
    
    def __repr__(self) -> str:
        return self.matrix.__repr__()
    
    def resize(self, *size: int):
        return resize(self.matrix, size)

    def pad(self, *size: int):
        # Calculate the amount of padding needed
        target_rows, target_cols = size
        rows, cols = self.matrix.shape
        pad_rows = target_rows - rows
        pad_cols = target_cols - cols

        # Pad only the bottom and right sides
        padded_matrix = np.pad(self.matrix, 
                            ((0, pad_rows), (0, pad_cols)), 
                            mode='constant', 
                            constant_values=0.5)
        return padded_matrix

    def plot(self, protein: 'Protein'):
        plt.title(f'{protein.name} Contact Map')
        plt.imshow(self.matrix, cmap = "Greys", vmin = self.matrix.min(), vmax = self.matrix.max())
        plt.show()
        plt.savefig('test.png')


class ContactMapPPI(ContactMap):
    def __init__(self, matrix: np.array):
        super().__init__(matrix)

    def aa(self, p1: 'Protein', p2: 'Protein') -> np.array:
        return self.matrix[:len(p1), :len(p1)]
    
    def bb(self, p1: 'Protein', p2: 'Protein') -> np.array:
        return self.matrix[len(p1):, len(p1):]
    
    def ab(self, p1: 'Protein', p2: 'Protein') -> np.array:
        return ContactMapPPI(self.matrix[:len(p1), len(p1):])

    def plot(self, ppi: 'ProteinProtein'):
        # Horizontal and vertical lines
        plt.vlines(x = len(ppi.p1), ymin = 0, ymax = len(ppi), colors = 'black')
        plt.hlines(y = len(ppi.p1), xmin = 0, xmax = len(ppi), colors = 'black')
        # Ticks
        lengths = np.cumsum([0, len(ppi.p1), len(ppi.p2)])
        ticks = (lengths[1:] + lengths[:-1])/2
        names = [ppi.p1.name, ppi.p2.name]
        plt.xticks(ticks, names)
        plt.yticks(ticks, names)
        # Plot
        plt.title(f'{ppi.p1.name}-{ppi.p2.name} Contact Map')
        plt.imshow(self.matrix, cmap = "Greys", vmin = self.matrix.min(), vmax = self.matrix.max())
        plt.show()
        plt.savefig('test.png')
    

if __name__ == "__main__":
    from src.entities.protein import Protein
    from src.entities.protein_protein import ProteinProtein
    import pandas as pd
    m = pd.read_csv('contacts.csv').to_numpy()
    cm = ContactMapPPI(m)
    p1 = Protein('O22456') # SEP3
    p2 = Protein('Q9FVC1') # SVP

    aa = cm.matrix[:len(p1), :len(p1)]
    bb = cm.matrix[len(p1):, len(p1):]
    ab = cm.matrix[:len(p1), len(p1):]
    ba = cm.matrix[len(p1):, :len(p1)]

    print(cm.plot(ProteinProtein(p1, p2)))
    

