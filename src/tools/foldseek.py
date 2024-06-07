# Built-in modules
import subprocess

# Third-party modules
import pandas as pd

# Custom libraries
from src.misc.logger import logger

class FoldSeek:

    def __init__(self, structure_directory: str):
        self.structure_directory = structure_directory

    def create_db(self) -> None:
        # Create FoldSeek database
        cmd = f'foldseek createdb {self.structure_directory} foldseek_db'
        subprocess.run(cmd, shell = True)

        # Generate and store the index on disk
        cmd = f'foldseek createindex foldseek_db tmp'
        subprocess.run(cmd, shell = True)

        # Logging
        logger.info('FoldSeek database created')

    def easy_search(self, pdb: str) -> pd.DataFrame:

        # Search for a PDB in the database
        cmd = f'foldseek easy-search {pdb} {self.structure_directory} aln tmp -v 2'
        subprocess.run(cmd, shell = True)

        # Convert output to pandas DataFrame
        df = pd.read_csv('aln', sep = '\t', header = None)
        df.columns = ['query','target','fident','alnlen','mismatch','gapopen','qstart','qend','tstart','tend','evalue','bits']

        # Remove tmp folder and aln file
        cmd = 'rm -r tmp'
        subprocess.run(cmd, shell = True)
        cmd = 'rm aln'
        subprocess.run(cmd, shell = True)

        # Logging
        logger.info(f'FoldSeek search for {pdb} completed')

        return df
        