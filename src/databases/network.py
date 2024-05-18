# Built-in modules
from collections import Counter

# Third-party modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor

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
    
    def interactors(self) -> list[Interactor]:
        '''
        Return list of network interactors.

        Returns
        -------
        list[Interactor]
            Interactors.
        '''
        unique_interactors = set(self.df['A']) | set(self.df['B'])
        return [Interactor.unpickle(uniprot_id) for uniprot_id in unique_interactors]
    
    def plot_per_taxa(self):
        '''
        Plot interactors and interactions per taxa.
        Interactions where the interactors are from the different species
        are not handled but considred as from one of the species.
        '''
        # Plot interactors per taxa
        interactors_per_taxa = Counter([str(interactor.taxon_id_to_name()) for interactor in self.interactors()])
        plt.bar(*zip(*interactors_per_taxa.most_common()))
        plt.xticks(rotation = 90)
        plt.xlabel('Species')
        plt.ylabel('Number of UniProt accessions')
        plt.title('Number of interactors per species')
        plt.tight_layout()
        plt.savefig(f'{path.PLOTS}/interactors_per_taxa_combined_databases.png')
        plt.clf()

        # Plot interactions per taxa
        get_taxon_name = lambda x: Interactor.unpickle(x).taxon_id_to_name()
        interactions_per_taxa = self.df['A'].apply(get_taxon_name).value_counts()
        sns.barplot(interactions_per_taxa)
        plt.xticks(rotation = 90)
        plt.xlabel('Species')
        plt.ylabel('Number of interactions')
        plt.title('Number of interactions per species')
        plt.tight_layout()
        plt.savefig(f'{path.PLOTS}/interactions_per_taxa_combined_databases.png')
        plt.clf()

if __name__ == '__main__':
    '''Test class.'''
    intact = Network(db = 'IntAct', version = '2024-02-14', type = 'MADS_vs_MADS')
    biogrid = Network(db = 'BioGRID', version = '4.4.233', type = 'MADS_vs_MADS')

    both = intact.merge(biogrid)
    both.plot_per_taxa()