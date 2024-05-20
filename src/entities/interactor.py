# Built-in modules
from __future__ import annotations
import os
from typing import Any, Generator

# Third-party modules
from tqdm import tqdm
# from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.misc import utils

class Interactor:

    # Initialize NCBI taxonomy tree
    # ncbi_tx = NcbiTx()

    def __init__(self, **kwargs: dict[str, Any]):
        # Modified with 'InterPro' class
        self.uniprot_id = kwargs.get('uniprot_id', '')
        self.seq = kwargs.get('seq', '')
        self.taxon_id = kwargs.get('taxon_id', '')
        self.domains = kwargs.get('domains', {
            'MADS-box': [],
            'K-box':    []
        })

        # Modified with 'parse_uniprot()' method
        self.uniprot_info = kwargs.get('uniprot_info', {
            'Section' : '',
            'Primary Accession' : '',
            'Secondary Accessions' : []
        })

    def __repr__(self) -> str:
        '''
        Prints instance attributes for debugging purposes.

        Returns
        -------
        str
            Instance attribute mapping.
        '''
        return str(self.__dict__)
    
    def taxon_id_to_name(self) -> str:
        '''
        Converts the NCBI taxon ID to the biological name it represents.
        For example, "3702" translates to "Arabidopsis thaliana"

        Returns
        -------
        str
            Taxon name.
        '''
        return Interactor.ncbi_tx.name(str(self.taxon_id))
    
    def pickle(self) -> None:
        '''
        Pickles the __dict__ of the Interactor object.
        Custom (un)pickling methods avoid excesive use of the utils 
        module and provides higher code abstraction. 
        '''
        filepath = f'{path.INTERACTORS}/{self.uniprot_id}.int'
        utils.pickling(data = self.__dict__, path = filepath)

    def unpickle(uniprot_id: str) -> Interactor:
        '''
        Unpickles __dict__ of an Interactor object. Returns it 
        reinstanciated. Custom (un)pickling methods avoid excesive use 
        of the utils module and provides higher code abstraction. 
        
        Parameters
        ----------
        uniprot_id : str
            UniProt ID corresponding to the file to be unpickled. Both 
            'O22456' and 'O22456.pkl' are managed.

        Returns
        -------
        Interactor
            Interactor object reinstanciated from unpickled __dict__ 
            object.
        '''
        uniprot_id = uniprot_id if uniprot_id.endswith('int') else uniprot_id + '.int'
        filepath = f'{path.INTERACTORS}/{uniprot_id}'
        interactor__dict__ = utils.unpickling(path = filepath)
        return Interactor(**interactor__dict__)
    
    @staticmethod
    def iterate_folder(folder: str, start:int = 0, limit: int = -1) -> Generator[Any, None, None]:
        for i, file in enumerate(tqdm(sorted(os.listdir(folder)))):
            if i == limit:
                break
            if i < start:
                continue
            
            yield Interactor.unpickle(file)

if __name__ == '__main__':
    '''Test class'''
    
    i = Interactor.unpickle('P48007')
    print(i)

