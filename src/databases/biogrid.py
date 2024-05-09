# Built-in modules
import os
import re
import subprocess
from io import StringIO

# Third-party modules
import pandas as pd
from multitax import NcbiTx

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger

class BioGRID:

    def __init__(self, version: str):
        self.version = version

    def download_files(self) -> None:
        '''
        Downloads and unzips BioGRID 'ALL' file from the specified 
        BioGRID version.
        '''

        # Set up output directory
        output_dir = f'{path.BIOGRID}/{self.version}'
        os.makedirs(output_dir, exist_ok = True)

        # Download 'ALL' file
        url = f'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{self.version}/'
        all_file = f'BIOGRID-ALL-{self.version}.tab3.zip'
        wget = f'wget {url}/{all_file} -P {output_dir} -q'
        subprocess.run(wget, shell = True)

        # Unzip file and remove compressed file
        unzip = f'unzip -qq {output_dir}/{all_file} -d {output_dir}'
        subprocess.run(unzip, shell = True)
        rm = f'rm {output_dir}/{all_file}'
        subprocess.run(rm, shell = True)

        # Logging
        logger.info(f'BioGRID {self.version} "ALL" file downloaded and unzipped')

    def reduce_to_plants(self) -> None:
        '''
        Reduces BioGRID 'ALL' file to only plant interactors to
        reduce computational burden when searching for UniProt IDs with
        grep in the whole file.
        '''

        # Set up file paths
        all_filepath = f'{path.BIOGRID}/{self.version}/BIOGRID-ALL-{self.version}.tab3.txt'
        plant_filepath = f'{path.BIOGRID}/{self.version}/BIOGRID-plants-{self.version}.tab3.txt'

        # Read BioGRID 'ALL' file as pandas DataFrame
        df = pd.read_csv(all_filepath, sep = '\t', dtype=str)

        # Filter only plant interactors
        ncbi_tx = NcbiTx()
        df_plants = df[df['Organism ID Interactor A'].apply(lambda x: 'Viridiplantae' in ncbi_tx.name_lineage(x))]

        # Save filtered file
        df_plants.to_csv(plant_filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'BioGRID {self.version} "ALL" file ({len(df)} PPIs) reduced to only plant interactions ({len(df_plants)} PPIs)')

    def _grep(self, uniprot_id: str) -> pd.DataFrame:
        '''
        Searches for a specific UniProt ID in the BioGRID 'plants' file and
        retrieves its interactions in BioGRID.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID to search for.

        Returns
        -------
        pd.DataFrame
            Interaction table of the given UniProt ID.
        '''

        # Set up file path
        file_path = f'{path.BIOGRID}/{self.version}/BIOGRID-plants-{self.version}.tab3.txt'

        # First line as columns
        cat = f'head -1 {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        columns = result.stdout.lstrip('#').rstrip().split('\t')

        # Search for UniProt ID in BioGRID 'ALL' file
        grep = f'grep {uniprot_id} {file_path}'
        result = subprocess.run(grep, shell = True, text = True, stdout = subprocess.PIPE)
        
        # Convert result to DataFrame
        if result.stdout == '':
            return pd.DataFrame(columns = columns)
        else:
            result = StringIO(result.stdout)
            return pd.read_csv(result, sep = '\t', names = columns)
        
    def mads_vs_all(self) ->  None:
        '''
        Searches for MADS interactors in the BioGRID 'plants' file and
        retrieves their interactions in BioGRID.
        '''
        # Initialize DataFrame
        mads_vs_all = pd.DataFrame()

        # Iterate over MADS interactors
        for interactor in utils.iterate_folder(path.INTERACTORS):
            # Search for UniProt ID in BioGRID 'plants' file
            df = self._grep(interactor.uniprot_id)
            # Append to DataFrame if not empty
            if not df.empty:
                mads_vs_all = pd.concat([mads_vs_all, df], ignore_index = True)
        
        # Save DataFrame
        filepath = f'{path.NETWORKS}/BioGRID_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in BioGRID {self.version} "plants" file -> dim({mads_vs_all.shape})')

    def mads_vs_mads(self) -> None:

        # Load MADS_vs_ALL DataFrame
        filepath = f'{path.NETWORKS}/BioGRID_{self.version}_MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # MADS UniProt IDs
        mads = set([interactor.uniprot_id for interactor in utils.iterate_folder(path.INTERACTORS)])

        # Concatenate UniProt IDs column A
        uniprot_columns_A = ['SWISS-PROT Accessions Interactor A', 'TREMBL Accessions Interactor A']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_A = mads_vs_all[uniprot_columns_A].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - mads) < len(x)
        mads_vs_mads_A = uniprot_ids_A.apply(is_there_mikc)

        # Concatenate UniProt IDs column B
        uniprot_columns_B = ['SWISS-PROT Accessions Interactor B', 'TREMBL Accessions Interactor B']
        concatenate = lambda row: '|'.join(row).replace('-|', '').rstrip('|-').split('|')
        uniprot_ids_B = mads_vs_all[uniprot_columns_B].apply(concatenate, axis = 1)
        is_there_mikc = lambda x: len(set(x) - mads) < len(x)
        mads_vs_mads_B = uniprot_ids_B.apply(is_there_mikc)

        # Filter MADS vs MADS interactions
        mads_vs_mads = mads_vs_all[mads_vs_mads_A & mads_vs_mads_B]

        # Save DataFrame
        filepath = f'{path.NETWORKS}/BioGRID_{self.version}_MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. MADS PPIs in BioGRID {self.version} "plants" file -> dim({mads_vs_mads.shape})')

if __name__ == '__main__':
    '''Test class'''
    biogrid = BioGRID('4.4.233')
    # biogrid.download_files()
    # biogrid.reduce_to_plants()
    biogrid.mads_vs_all()
    biogrid.mads_vs_mads()