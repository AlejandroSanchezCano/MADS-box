# Built-in modules
from __future__ import annotations
from typing import Generator
import re
import os


# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.tools.esm_fold import ESMFold
from src.tools.foldseek import FoldSeek
from src.entities.interactor import Interactor
from src.entities.contact_map import ContactMap

class Protein:

    bible = {
        'AtAP1' : 'P35631'
    }

    def __init__(self, name: str):
        # Instantiate from folder (from __dict__)
        try:
            __dict__ = utils.unpickling(f'{path.LITERATUREMINING}/Proteins/{name.split(".")[0]}.prot')
            for attribute, value in __dict__.items():
                setattr(self, attribute, value)
        
        # New instantiation
        except FileNotFoundError:
            # From name
            self.name = name
            self.uniprot_id, *self.mutations = name.split('_')

            # From Interactor
            interactor = Interactor.unpickle(self.uniprot_id)
            self.seq = self._mutate() if self.mutations else interactor.seq
            self.taxon_id = interactor.taxon_id
            self.domains = {
                'MADS-box': interactor.domains['MADS-box'][0],
                'K-box': interactor.domains['K-box'][0],
            }
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def __len__(self) -> int:
        return len(self.seq)
    
    def save(self) -> None:
        utils.pickling(self.__dict__, f'{path.PROTEIN}/{self.name}.prot')

                            ### MUTATION ###
                            
    # TODO -> double mutations are not done over the orginal sequence
    def _mutate(self) -> str:
        # Wild-type
        seq = Protein(self.uniprot_id).seq

        # Iterate over mutations
        for mutation in self.mutations:

            # MADS deletion
            if mutation == 'IKC':
                seq = Protein(self.uniprot_id).ikc
            
            # Domain swapping
            elif '[' in mutation:
                # Parse mutation
                domains, origin = re.findall(r'^([MIKC]*)\[([a-zA-Z0-9]*)\]', mutation)[0]
                # Protein where swapping will occurr
                prot_wt = Protein(self.uniprot_id)
                # Sequence to extract domain
                prot_mut = Protein(Protein.bible[origin])
                # Swap
                if domains == 'I':
                    seq = prot_wt.seq.replace(prot_wt.i, prot_mut.i)
                else:
                    raise NotImplementedError(f'Domain swapping for {domains} is not implemented')
            
            # Regular mutation
            else:
                
                # Parse mutation
                wt, index, mut = re.findall(r'^([A-Z]*)([0-9]*)([A-Z]*)', mutation)[0]
                # Validate mutation
                if seq[int(index) - 1 : int(index) - 1 + len(wt)] != wt:
                    raise ValueError(f'Mutation {mutation} is not valid for sequence {seq}') 
                # Mutate
                before = seq[:int(index) - 1]
                after = seq[int(index) + len(wt) - 1:]
                seq = before + mut + after
        
        return seq
    
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
    
                            ### ITERATORS ###
    
    @staticmethod
    def iterate() -> Generator[Protein, None, None]:
        for protein_file in tqdm(sorted(os.listdir(f'{path.LITERATUREMINING}/Proteins'))):
            yield Protein(protein_file)

                            ### ADDITIONS ###
    
    @staticmethod
    def add_esm2() -> None:
        esm = ESMFold()
        esm.performance_optimizations()
        for protein in Protein.iterate():
            if protein.pdb and protein.plddt and protein.pae and protein.contacts:
                continue
            esm.fold([protein.mik])
            protein.pdb = esm.pdb
            protein.plddt = esm.plddt
            protein.pae = esm.pae
            protein.contacts = ContactMap(esm.contacts)
            protein.save()

    @staticmethod
    def add_foldseek() -> None:
        foldseek = FoldSeek(f'{path.ARABIDOPSIS}/ESMFold')
        for protein in Protein.iterate():
            if protein.most_similar:
                continue
            aln = foldseek.easy_search(f'{path.LITERATUREMINING}/PDBs/{protein.name}.pdb')
            print(aln)
            protein.most_similar = aln.loc[aln['evalue'].idxmin(), 'target']
            print(protein.name, protein.most_similar)
            
            protein.save()

if __name__ == '__main__':
    # from collections import Counter
    # ms = Counter([p.most_similar for p in Protein.iterate()])
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # df = pd.DataFrame.from_dict(ms, orient='index', columns=['values'])
    # # Reset index to make 'one', 'two', etc., regular columns
    # df.reset_index(inplace=True)
    # df = df.sort_values(by='values', ascending=False)
    # # Step 2: Use Seaborn to create a boxplot
    # plt.figure(figsize=(10, 10))  # Optional: Adjust figure size
    # sns.barplot(y=df['index'], x=df['values'], orient = 'y')
    # plt.title('Number of most similar Arabidopsis MKC using Foldseek')
    # plt.xlabel('# of Occurrences')
    # plt.ylabel('Arabidopsis MIKC')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('.')

    p = Protein('Q9SI38_N60_Y103H_A191V')
    print(p.domains)
    print(len(p))
