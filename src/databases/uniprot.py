# Built-in modules
import os
import subprocess

# Third-party modules
from bioservices.uniprot import UniProt as UniProtAPI

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger
from src.entities.interactor import Interactor

class UniProt:

    # Initialize UniProt API
    uniprot_api = UniProtAPI(verbose = False)

    def fill_sequences(self) -> None:
        '''
        Download the sequences of all the MIKC proteins found with 
        InterPro from the UniProt database and add them to the 
        Interactor objects.
        '''
        for interactor in Interactor.iterate_folder(path.INTERACTORS):
            # Download FASTA from UniProt database
            cmd = f'curl "https://rest.uniprot.org/uniprotkb/{interactor.uniprot_id}.fasta"'
            response = subprocess.run(cmd, capture_output = True, text = True, shell = True).stdout
            # Add sequence to Interactor object
            interactor.seq = utils.read_fasta_str(response)[1] 
            # Save Interactor object
            utils.pickling(data = interactor.__dict__, path = f'{path.INTERACTORS}/{interactor.uniprot_id}.int')
        
        # Logging
        n_interactors = len(os.listdir(path.INTERACTORS))
        logger.info(f'{n_interactors} sequences have been obtained from UniProt')

    def __parse_uniprot_entry(self, interactor: Interactor) -> None:
        '''
        Uses BioServices' UniProt API wrapper to retrieve relevant 
        information, stored in the "uniprot_info()" attribute:
        - self.uniprot_info['Section'] -> Swiss-Prot or TrEMBL.
        - self.uniprot_info['Primary Accession'] -> entry's primary 
        accession
        - self.uniprot_info['Secondary Accessions'] -> if applicable, 
        entry's secondary accessions.

        Same time complexity doing calls with individual or multiple 
        UniProt IDs, so I opted to individual calls to match abstraction 
        levels.

        Parameters
        ----------
        interactor : Interactor
            Interactor object to fill with UniProt information.
        '''
        # Retrieve UniProt data
        entry_json = UniProt.uniprot_api.retrieve(
            uniprot_id = interactor.uniprot_id, 
            frmt = 'json',
            database = 'uniprot'
            )

        # Parse relevant information
        interactor.taxon_id = entry_json['organism']['taxonId']
        interactor.uniprot_info['Section'] = 'TrEMBL' if entry_json['entryType'].endswith('(TrEMBL)') else 'Swiss-Prot'
        interactor.uniprot_info['Primary Accession'] = entry_json.get('primaryAccession')
        interactor.uniprot_info['Secondary Accessions'] = entry_json.get('secondaryAccessions')
        
        # Logging
        logger.info(f'{interactor.uniprot_id} has {interactor.uniprot_info=}')

    def fill_metadata(self) -> None:
        '''
        Download the metadata of all the MIKC proteins found with 
        InterPro from the UniProt database and add them to the 
        Interactor objects.
        '''
        for interactor in Interactor.iterate_folder(path.INTERACTORS):
            # Download metadata from UniProt database
            self.__parse_uniprot_entry(interactor)

            # Save Interactor object
            utils.pickling(data = interactor.__dict__, path = f'{path.INTERACTORS}/{interactor.uniprot_id}.int')
        
        # Logging
        n_interactors = len(os.listdir(path.INTERACTORS))
        logger.info(f'{n_interactors} metadata have been obtained from UniProt')

    def download_structures(self) -> None: 
        '''
        Download AlphaFold structures (14106) of all the MIKC proteins 
        found with InterPro (17090) from the AlphaFold database.
        '''
        for interactor in Interactor.iterate_folder(path.INTERACTORS):
            # Download response from AlphaFold database
            structure_path = f'{path.UNIPROT_STRUCTURES}/{interactor.uniprot_id}.pdb'
            cmd = f'curl "https://alphafold.ebi.ac.uk/files/AF-{interactor.uniprot_id}-F1-model_v4.pdb"'
            response = subprocess.run(cmd, capture_output = True, text = True, shell = True).stdout

            # Only save the non-empty responses (> 200 characters)
            if len(response) > 200:
                with open(structure_path, 'w') as handle:
                    handle.write(response)

        # Logging
        n_pdb_files = len(os.listdir(path.UNIPROT_STRUCTURES))
        n_interactors = len(os.listdir(path.INTERACTORS))
        logger.info(f'{n_pdb_files} AlphaFold files have been downloaded out of {n_interactors} MIKC proteins')

if __name__ == '__main__':
    '''Test class'''
    uniprot = UniProt()
    #uniprot.fill_sequences()
    #uniprot.fill_metadata()
    uniprot.download_structures()