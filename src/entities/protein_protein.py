# Built-in modules
from __future__ import annotations
import os
from typing import Generator

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger
from src.tools.esm_fold import ESMFold
from src.entities.protein import Protein
from src.entities.contact_map import ContactMapPPI

class ProteinProtein:

    pickle_path = path.PROTEINPROTEIN

    def __init__(self, p1: str | Protein, p2: str | Protein):
        # Instantiate from folder (from __dict__)
        try:
            name1 = p1.name if isinstance(p1, Protein) else p1
            name2 = p2.name if isinstance(p2, Protein) else p2
            __dict__ = utils.unpickling(f'{ProteinProtein.pickle_path}/{name1}-{name2}.ppi')
            for attribute, value in __dict__.items():
                setattr(self, attribute, value)
            self.p1 = Protein(self.p1)
            self.p2 = Protein(self.p2)
        
        # New instantiation
        except FileNotFoundError:
            self.p1 = Protein(p1) if isinstance(p1, str) else p1
            self.p2 = Protein(p2) if isinstance(p2, str) else p2
            self.interaction = None
            self.origin = None
            self.pdb = None
            self.plddt = None
            self.pae = None
            self.contacts = None

    def __len__(self) -> int:
        return len(self.p1) + len(self.p2)
    
    def __repr__(self) -> str:
        return str(self.__dict__)

    def save(self) -> None:
        self.p1 = self.p1.name
        self.p2 = self.p2.name
        utils.pickling(self.__dict__, f'{ProteinProtein.pickle_path}/{self.p1}-{self.p2}.ppi')

                            ### ITERATION ###

    @staticmethod
    def iterate() -> Generator[ProteinProtein, None, None]:
        for ppi_file in tqdm(sorted(os.listdir(path.PROTEINPROTEIN))):
            p1, p2 = ppi_file.split('.')[0].split('-')	
            yield ProteinProtein(p1, p2)

    @staticmethod
    def iterate_paper() -> Generator[ProteinProtein, None, None]:
        for ppi_file in tqdm(sorted(os.listdir(path.PROTEINPROTEIN))):
            p1, p2 = ppi_file.split('.')[0].split('-')
            ppi = ProteinProtein(p1, p2)
            if ppi.origin not in ['IntAct', 'BioGRID', 'PlaPPISite']:
                yield ppi

    @staticmethod
    def iterate_paper_mikc() -> Generator[ProteinProtein, None, None]:
        for ppi_file in tqdm(sorted(os.listdir(path.PROTEINPROTEIN))):
            p1, p2 = ppi_file.split('.')[0].split('-')
            ppi = ProteinProtein(p1, p2)
            if ppi.origin not in ['IntAct', 'BioGRID', 'PlaPPISite'] and 'IKC' not in ppi.p1.name and not 'IKC' in ppi.p2.name:
                yield ppi

                            ### ADDITIONS ###                                   
    @staticmethod
    def add_esmfold():
        # Prepare ESMFold 
        esm = ESMFold()
        esm.performance_optimizations()
        # Iterate over ProteinProtein instances
        for ppi in ProteinProtein.iterate():
            # Skip if already processed
            if ppi.pdb:
                continue
            # Logging
            logger.info(f'Processing {ppi.p1.name}-{ppi.p2.name}')
            # Process full protein if IKC
            seq1 = ppi.p1.seq if 'IKC' in ppi.p1.name else ppi.p1.mik
            seq2 = ppi.p2.seq if 'IKC' in ppi.p2.name else ppi.p2.mik
            # Fold
            esm.fold([seq1, seq2])
            # Add attributes
            ppi.pdb = esm.pdb
            ppi.plddt = esm.plddt
            ppi.pae = esm.pae
            ppi.contacts = ContactMapPPI(esm.contacts)
            # Save
            ppi.save()

if __name__ == '__main__':
    ProteinProtein.add_esmfold()
