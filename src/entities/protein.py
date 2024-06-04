# Built-in modules
import re

# Custom modules
from src.entities.interactor import Interactor

class Protein:

    def __init__(self, uniprot_id: str, mutation: str = None):

        # Instantiation
        self.uniprot_id = uniprot_id
        self.mutation = mutation

        # From Interactor
        interactor = Interactor.unpickle(self.uniprot_id)
        self.seq = self._mutate() if mutation else interactor.seq
        self.taxon_id = interactor.taxon_id
        self.domains = {
            'MADS-box': interactor.domains['MADS-box'][0],
            'K-box': interactor.domains['K-box'][0],
        }
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
                            ### MUTATION ###

    def _mutate(self) -> str:
        # Wild-type
        seq = Protein(self.uniprot_id).seq

        # MADS deletion
        if self.mutation == 'IKC':
            return Protein(self.uniprot_id).ikc
        
        # Domain swapping
        elif '[' in self.mutation:
            # Parse mutation
            domains, origin = re.findall(r'^([MIKC]*)\[([a-zA-Z0-9]*)\]', self.mutation)[0]
            # Protein where swapping will occurr
            prot_wt = Protein(self.uniprot_id)
            # Sequence to extract domain
            prot_mut = Protein(origin)
            # Swap
            if domains == 'I':
                return prot_wt.seq.replace(prot_wt.i, prot_mut.i)
            else:
                raise NotImplementedError(f'Domain swapping for {domains} is not implemented')
        
        # Regular mutation
        else:
            # Parse mutation
            wt, index, mut = re.findall(r'^([A-Z]*)([0-9]*)([A-Z]*)', self.mutation)[0]
            # Validate mutation
            if seq[int(index) - 1 : len(wt)] != wt:
                raise ValueError(f'Mutation {self.mutation} is not valid for sequence {seq}') 
            # Mutate
            before = seq[:int(index) - 1]
            after = seq[int(index) + len(wt) - 1:]
            return before + mut + after
    
                        ### DOMAIN ARCHITECTURE ###

    @property
    def m(self) -> str:
        return self.seq[self.domains['MADS-box'][0]:self.domains['MADS-box'][1]]
    
    @property
    def i(self) -> str:
        return self.seq[self.domains['MADS-box'][1]:self.domains['K-box'][0]]

    @property
    def k(self) -> str:
        return self.seq[self.domains['K-box'][0]:self.domains['K-box'][1]]
    
    @property
    def c(self) -> str:
        return self.seq[self.domains['K-box'][1]:]
    
    @property
    def mi(self) -> str:
        return self.m + self.i
    
    @property
    def mik(self) -> str:
        return self.m + self.i + self.k
    
    @property
    def ik(self) -> str:
        return self.i + self.k

    @property
    def ikc(self) -> str:
        return self.i + self.k + self.c

if __name__ == '__main__':

    prot = Protein('Q8LLR0')
    print(prot)
    prot = Protein('Q8LLR0', 'I[Q8LLR0]')
    print(prot)