
# Third-party modules
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from src.entities.protein import Protein

class ContactMap():

    def __init__(self, contact_map: np.array):
        self.contact_map = contact_map

    def __str__(self):
        return self.contact_map
    
    def __repr__(self):
        return self.contact_map
    
    def plot(self, protein: Protein):
        plt.title(f'{protein.name} Contact Map')
        plt.imshow(self.contact_map, cmap = "Greys", vmin = 0, vmax = 1)
        plt.show()
        plt.savefig('.png')

if __name__ == "__main__":
    from src.entities.protein import Protein
    p = Protein('P48007')
    cm = ContactMap(p.contacts).plot(p)
