# Built-in modules
from collections import Counter

# Third-party modules
import pandas as pd
import seaborn as sns
from tqdm import tqdm 
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor

class Network:

    def __init__(
            self,
            db: str = None,
            version: str = None,
            type: str = None, 
            standarized: bool = True
            ):
        
        standarized = '_standarized' if standarized else ''
        version = f'_{version}' if version else ''
        filepath = f'{path.NETWORKS}/{db}{version}_{type}{standarized}.tsv'
        self.df = pd.read_csv(filepath, sep = '\t') if db and type else None

    def __repr__(self) -> str:
        return self.df.__repr__()

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Network':
        network = cls()
        network.df = df
        return network

    def merge(self, other) -> pd.DataFrame:
        df_df = pd.concat([self.df, other.df], ignore_index = True)
        return Network.from_df(df_df.drop_duplicates('A-B'))
    
    def interactors(self, df: pd.DataFrame = None) -> list[Interactor]:
        df = self.df if df is None else df
        unique_interactors = set(df['A']) | set(df['B'])
        return [Interactor.unpickle(uniprot_id) for uniprot_id in unique_interactors]
    
    @property
    def species(self) -> list[int]:
        speciesA = self.df['Species_A'].unique()
        speciesB = self.df['Species_B'].unique()
        return list(set(speciesA) | set(speciesB))
    
    def plot_per_taxa(self) -> None:
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

    def add_negatives_per_species(self) -> 'Network':
        # Initialize dictionary to store negatives per species
        df_negatives = {
            'A': [],
            'B': [],
            'A-B': [],
            'Species_A': [],
            'Species_B': []
        }
        # Get interactors per species
        for species in tqdm(self.species):
            df = self.df[self.df.apply(lambda x: x['Species_A'] == species and x['Species_B'] == species, axis = 1)]
            interactors = self.interactors(df)
            # Add negatives
            for int1 in interactors:
                for int2 in interactors:
                    name = '-'.join(sorted([int1.uniprot_id, int2.uniprot_id]))
                    if name not in df['A-B']:
                        df_negatives['A'].append(int1.uniprot_id)
                        df_negatives['B'].append(int2.uniprot_id)
                        df_negatives['A-B'].append(f'{int1.uniprot_id}-{int2.uniprot_id}')
                        df_negatives['Species_A'].append(species)
                        df_negatives['Species_B'].append(species)
        
        # Add interaction column
        self.df['Interaction'] = 1
        df_negatives = pd.DataFrame(df_negatives)
        df_negatives['Interaction'] = 0

        # Add negatives to the dataframe
        df = pd.concat([self.df, df_negatives], ignore_index = True)
        return Network.from_df(df)

if __name__ == '__main__':
    '''Test class.'''
    intact = Network(db = 'IntAct', version = '2024-02-14', type = 'MADS_vs_MADS')
    biogrid = Network(db = 'BioGRID', version = '4.4.233', type = 'MADS_vs_MADS')

    print(intact.add_negatives_per_species())