# Built-in modules

# Third-party modules
import numpy as np
from Bio import PDB
import seaborn as sns
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger

class Structure:

    # Genetic code
    three_to_one = {
        'ALA':'A', 'VAL':'V', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F',
        'TYR':'Y', 'TRP':'W', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 
        'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'CYS':'C',
        'GLY':'G', 'PRO':'P'
        }

    def __init__(self, uniprot_id: str):
        # Instantiate from 'Structures' folder (from __dict__)
        try:
            structure__dict__ = utils.unpickling(f'{path.STRUCTURES}/{uniprot_id.split(".")[0]}.str')
            for attribute, value in structure__dict__.items():
                setattr(self, attribute, value)
        
        # Instantiate from 'PDBs' folder (parse PDB)
        except FileNotFoundError:
            self.uniprot_id = uniprot_id
            self.path = f'{path.UNIPROT_PDB}/{self.uniprot_id}.pdb'
            self.structure = self.structure()
            self.seq3 = self.seq3()
            self.seq1 = self.seq1()
            self.plddt = self.plddt()
            self.ss = self.dssp()
              
    def structure(self) -> PDB.Structure.Structure:
        '''
        Parses PDB file and returns a BioPython Structure object.

        Returns
        -------
        PDB.Structure.Structure
            BioPython PDB.Structure.Structure object.
        '''
        return PDB.PDBParser().get_structure(id = self.uniprot_id, file = self.path)
    
    def seq3(self) -> str:
        '''
        Returns a string with the three-letter amino acid sequence.

        Returns
        -------
        str
            Three-letter amino acid sequence.
        '''
        return '-'.join([residue.resname for residue in self.structure.get_residues()])
    
    def seq1(self) -> str:
        '''
        Returns a string with the one-letter amino acid sequence.

        Returns
        -------
        str
            One-letter amino acid sequence.
        '''
        return ''.join([Structure.three_to_one[residue.resname] for residue in self.structure.get_residues()])
    
    def plddt(self) -> np.ndarray:
        '''
        Returns an array with the per-residue PLDDT values.

        Returns
        -------
        np.ndarray
            PLDDT values.
        '''
        return np.array([next(residue.get_atoms()).bfactor for residue in self.structure.get_residues()])
    
    def dssp(self) -> str:
        '''
        Infers the secondary structure using DSSP from Bio.

        Returns
        -------
        str
            Secondary structure with DSSP code.
        '''
        dssp = PDB.DSSP(self.structure[0], self.path, dssp='mkdssp')
        _, _, secondary_structure, *_ = zip(*dssp.property_list)
        return ''.join(secondary_structure)
    
    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)
    
    def pLDDT_over_seq(self):
        '''
        Builds a scatterplot from the pLDDT values with respect to the
        sequence index and colors the background according to each pLDDT 
        quality class range. At the bottom, draws a rugplot (1-D) of the 
        pLDDT values (as in InterPro).
        '''
        # Scatterplot and rugplot
        y = self.pLDDT
        x = list(range(len(y)))
        hue = ['Very high' if i >= 90 else 'Condifent' if i >= 70 else 'Low' if i >= 50 else 'Very low' for i in y]
        plddt_palette = {'Very high': '#0053D6', 'Condifent':'#65CBF3', 'Low':'#FFDB13', 'Very low':'#FF7D45'}
        fig = sns.scatterplot(x=x, y=y, color='black')
        sns.rugplot(x=x, hue = hue, palette = plddt_palette, legend = False, linewidth = 2, expand_margins=True)        

        # Horizontal spans
        ax = plt.gca()
        ax.axhspan(90, 100, alpha = 0.5, color = '#0053D6')
        ax.axhspan(70, 90, alpha = 0.5, color = '#65CBF3')
        ax.axhspan(50, 70, alpha = 0.5, color = '#FFDB13')
        ax.axhspan(0, 50, alpha = 0.5, color = '#FF7D45')
        ax.set_ylim(0, 100)

        # Axis labels
        fig.set_xlabel('Residue index')
        fig.set_ylabel('pLDDT')

        # Show plot
        plt.show() 

    def pickle(self) -> None:
        '''
        Pickles the __dict__ of the Structure object.
        Custom (un)pickling methods avoid excesive use of the utils 
        module and provides higher code abstraction. 
        '''
        filepath = f'{path.STRUCTURES}/{self.uniprot_id}.str'
        utils.pickling(data = self.__dict__, path = filepath)
        
if __name__ == '__main__':
    structure = Structure(uniprot_id = 'A0A0A0KC22')

