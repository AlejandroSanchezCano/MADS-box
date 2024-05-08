# Built-in modules
import os
import subprocess
from io import StringIO

# Third-party modules
import pandas as pd
from tqdm import tqdm

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
        logger.info(f'BioGRID {self.version} files downloaded and unzipped')

    def _grep(self, uniprot_id: str) -> pd.DataFrame:
        '''
        Searches for a specific UniProt ID in the BioGRID 'ALL' file and
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
        file_path = f'{path.BIOGRID}/{self.version}/BIOGRID-ALL-{self.version}.tab3.txt'

        # First line as columns
        cat = f'head -1 {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        columns = result.stdout.lstrip('#').rstrip().split('\t')
        print(columns)

        # Search for UniProt ID in BioGRID 'ALL' file
        grep = f'grep {uniprot_id} {file_path}'
        result = subprocess.run(grep, shell = True, text = True, stdout = subprocess.PIPE)
        
        # Convert result to DataFrame
        if result.stdout == '':
            return pd.DataFrame(columns = columns)
        else:
            result = StringIO(result.stdout)
            df = pd.read_csv(result, sep = '\t', header = None)
            df.columns = columns
            return df

    def test(self):
        for interactor in utils.iterate_folder(path.INTERACTORS):
            df = biogrid._grep(interactor.uniprot_id)

    def test2(self):
        df = biogrid.full()
        from multitax import NcbiTx
        ncbi_tx = NcbiTx()
        df = df[df['Organism ID Interactor A'].apply(lambda x: 'Viridiplantae' in ncbi_tx.name_lineage(x))]

        print(df['Organism ID Interactor A'].unique())
        df.to_csv('out.csv', index=False, sep = '\t') 

    def test3(self):
        for interactor in utils.iterate_folder(path.INTERACTORS, start = 12000):
            grep = f'grep {interactor.uniprot_id} out.csv'
            result = subprocess.run(grep, shell = True, text = True, stdout = subprocess.PIPE)
            if result.stdout != '':
                print(interactor.uniprot_id)


    def full(self): #(2685273, 37) #83004
        file_path = f'{path.BIOGRID}/{self.version}/BIOGRID-ALL-{self.version}.tab3.txt' 

        cat = f'head -1 {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        columns = result.stdout.lstrip('#').rstrip().split('\t')

        file_path = f'{path.BIOGRID}/{self.version}/BIOGRID-ALL-{self.version}.tab3.txt' 
        cat = f'cat {file_path}'
        result = subprocess.run(cat, shell = True, text = True, stdout = subprocess.PIPE)
        df = pd.read_csv(StringIO(result.stdout), sep = '\t', names = columns, dtype=str)
        return df

if __name__ == '__main__':
    biogrid = BioGRID('4.4.233')
    #biogrid.download_files()
    #df = biogrid._grep('O22456')
    #print(df['Organism ID Interactor A'])
    #biogrid.test()
    df = biogrid.test3()