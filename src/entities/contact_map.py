
# Third-party modules
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class ContactMap():

    def __init__(self, matrix: np.array):
        self.matrix = matrix
    
    def __repr__(self) -> str:
        return self.matrix.__repr__()

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
        return ContactMapPPI(self.matrix[:len(p1), :len(p1)])
    
    def bb(self, p1: 'Protein', p2: 'Protein') -> np.array:
        return ContactMapPPI(self.matrix[len(p1):, len(p1):])
    
    def ab(self, p1: 'Protein', p2: 'Protein') -> np.array:
        return ContactMapPPI(self.matrix[:len(p1), len(p1):])

    def i(self, p1: 'Protein', p2: 'Protein') -> np.array:
        p1_i_start = p1.domains['MADS-box'][1] - 10
        p1_i_end = p1.domains['K-box'][1]
        p2_i_start = p2.domains['MADS-box'][1] - 10
        p2_i_end = p2.domains['K-box'][1]
        return ContactMapPPI(self.matrix[p1_i_start:p1_i_end, p2_i_start:p2_i_end])

    def m(self, p1: 'Protein', p2: 'Protein') -> np.array:
        p1_m_start = p1.domains['MADS-box'][0]
        p1_m_end = p1.domains['MADS-box'][1]
        p2_m_start = p2.domains['MADS-box'][0]
        p2_m_end = p2.domains['MADS-box'][1]
        return ContactMapPPI(self.matrix[p1_m_start:p1_m_end, p2_m_start:p2_m_end])

    def k(self, p1: 'Protein', p2: 'Protein') -> np.array:
        p1_k_start = p1.domains['K-box'][0]
        p1_k_end = p1.domains['K-box'][1]
        p2_k_start = p2.domains['K-box'][0]
        p2_k_end = p2.domains['K-box'][1]
        return ContactMapPPI(self.matrix[p1_k_start:p1_k_end, p2_k_start:p2_k_end])

    def plot(self, ppi: 'ProteinProtein', name: str = None):
        # Determine protein lengths
        p1_length = len(ppi.p1.mik) if 'IKC' not in ppi.p1.name else len(ppi.p1)
        p2_length = len(ppi.p2.mik) if 'IKC' not in ppi.p2.name else len(ppi.p2)
        ppi_length = p1_length + p2_length
        # Horizontal and vertical lines
        plt.vlines(x = p1_length, ymin = 0, ymax = ppi_length, colors = 'black')
        plt.hlines(y = p1_length, xmin = 0, xmax = ppi_length, colors = 'black')
        # Ticks
        lengths = np.cumsum([0, p1_length, p2_length])
        ticks = (lengths[1:] + lengths[:-1])/2
        names = [ppi.p1.name, ppi.p2.name]
        plt.xticks(ticks, names)
        plt.yticks(ticks, names)
        # Plot
        plt.title(f'{ppi.p1.name}-{ppi.p2.name} Contact Map')
        plt.imshow(self.matrix, cmap = "Greys", vmin = self.matrix.min(), vmax = self.matrix.max())
        plt.show()
        plt.savefig(f'{name if name else "contactmapppi"}.png')
    

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
    

