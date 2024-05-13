# Built-in modules
from collections import defaultdict

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.misc.logger import logger

class Network:

    def __init__(
            self,
            db: str = 'IntAct',
            version: str = '2024-02-14',
            type: str = 'MADS_vs_MADS', 
            standarized: bool = True
            ):
        
        standarized = '_standarized' if standarized else ''
        filepath = f'{path.NETWORKS}/{db}_{version}_{type}{standarized}.tsv'
        self.df = pd.read_csv(filepath, sep = '\t')

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        '''
        Factory method to create a Network object from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the network data.

        Returns
        -------
        Network
            Network object.
        '''
        network = cls()
        network.df = df
        return network

    def merge(self, other) -> pd.DataFrame:
        '''
        Merge two networks.

        Returns
        -------
        pd.DataFrame
            Merged network.
        '''
        df_df = pd.concat([self.df, other.df], ignore_index = True)
        return Network.from_df(df_df.drop_duplicates('A-B'))
    
    def statistics(self):
        n_interactions = len(self.df)
        n_interactors = len(set(self.df['A']) | set(self.df['B']))

        return n_interactions, n_interactors
    
if __name__ == '__main__':
    '''Test class.'''
    intact = Network(db = 'IntAct', version = '2024-02-14', type = 'MADS_vs_MADS')
    biogrid = Network(db = 'BioGRID', version = '4.4.233', type = 'MADS_vs_MADS')

    both = intact.merge(biogrid)
    print(both.statistics())