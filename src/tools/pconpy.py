# Built-in modules
import os
from typing import Literal

# Third-party modules
import numpy as np

# Custom modules
from src.pconpy import pconpy

class PConPy:

    def __init__(
            self, 
            map_type: Literal['cmap', 'dmap'], 
            pdb: str,
            measure: Literal['CA', 'CB', 'cmass', 'sccmass', 'minvdw'],
            distance: float = None
            ):
        
        self.map_type = map_type
        self.pdb = pdb
        self.measure = measure
        self.distance = distance

    def map(self) -> np.array:

        # Write the structure to a temporary PDB file
        with open('temp.pdb', 'w') as temp_pdb_file:
            temp_pdb_file.write(self.pdb + '\n')

        # Generate the map
        residues = pconpy.get_residues('temp.pdb', chain_ids=None)
        matrix = pconpy.calc_dist_matrix(residues = residues, measure = self.measure, dist_thresh = self.distance)

        # Clean up the temporary PDB file
        os.remove('temp.pdb')

        return matrix
    
    def standarize(self, matrix: np.array) -> np.array:
        return (matrix.max() - matrix) / matrix.max()

if __name__ == '__main__':
    from src.entities.protein_protein import ProteinProtein
    from src.entities.protein import Protein
    from src.entities.contact_map import ContactMapPPI
    p = Protein('A0A0E4AZI0')
    ppi = ProteinProtein(p, p)

    for measure in ['CA', 'CB', 'cmass', 'sccmass', 'minvdw']:
        pcon = PConPy('cmap', ppi.pdb, measure)
        matrix = pcon.map()
        cmap = pcon.standarize(matrix)
        ContactMapPPI(cmap).plot(ppi, measure)




    