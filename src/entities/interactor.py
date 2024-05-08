# Built-in modules
from __future__ import annotations
from typing import Any

# Third-party modules
from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.misc import utils

class Interactor:

    # Initialize NCBI taxonomy tree
    #ncbi_tx = NcbiTx()

    def __init__(self, **kwargs: dict[str, Any]):
        # Modified with 'InterPro' class
        self.uniprot_id = kwargs.get('uniprot_id', '')
        self.seq = kwargs.get('seq', '')
        self.taxon_id = kwargs.get('taxon_id', '')
        self.domains = kwargs.get('domains', {
            'MADS-box': [],
            'K-box':    []
        })

        # Modified with 'Database' class and its children
        self.biogrid_id = kwargs.get('biogrid_id', '')
        self.string_id = kwargs.get('string_id', '')
        self.intact_id = kwargs.get('intact_id', '')
        self.plappisite_id = kwargs.get('plappisite_id', '')

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

    def __eq__(self, other: Interactor) -> bool:
        '''
        Overrides the equal (==) operator by considering that two 
        Interactor objects are equal if they have the same UniProt ID. 
        Also, this is necessary along with the __hash__ dunder method 
        for the functionality of sets.

        Parameters
        ----------
        other : Interactor
            Other Interactor object.

        Returns
        -------
        bool
            Whether the same UniProt ID is shared by the Interactor 
            objects.
        '''
        return self.uniprot_id == other.uniprot_id
    
    def __hash__(self) -> int:
        '''
        Overrides the hash function by hashing the UniProt ID string. 
        Also, this is necessary along with the __eq__ dunder method for
        the functionality 
        of sets.

        Returns
        -------
        int
            Hash.
        '''
        return hash(self.uniprot_id)
    
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

if __name__ == '__main__':
    '''Test class'''
    
    i = Interactor.unpickle('A0A0A0KC22')
    print('File not found')

