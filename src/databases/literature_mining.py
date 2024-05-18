# Built-in modules
import os
from collections import defaultdict

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
from src.misc import path

class LiteratureMining:

    def __get_sheets(self) -> tuple[dict[str, pd.DataFrame], dict[str, list[pd.DataFrame]]]:
        '''
        Process the excel files in the LiteratureMining directory and 
        return:
        - 'Guidelines.xlsx' file as a sheet name to sheet content 
        dictionary
        - Paper Excels as a file name to list of data frame sheets 
        dictionary

        Returns
        -------
        tuple[dict[str, pd.DataFrame], dict[str, list[pd.DataFrame]]]
            Guidelines and papers data.
        '''
        # Guidelines file
        guidelines = pd.read_excel(f'{path.LITERATUREMINING}/Guidelines.xlsx', sheet_name = None)

        # Other files
        excel_files = sorted(os.listdir(path.LITERATUREMINING))
        excel_files.remove('Guidelines.xlsx')
        open_file = lambda x: pd.read_excel(f'{path.LITERATUREMINING}/{x}', na_values = ['ND'], sheet_name = None)
        filename2df = {filename : list(open_file(filename).values())for filename in excel_files}
        
        return guidelines, filename2df
    
    def interactions_per_file(self) -> np.ndarray:
        '''
        Get the number of interactions per file.

        Returns
        -------
        np.ndarray
            Number of interactions per file.
        '''
        # Iterate over files
        excel_files = self.__get_sheets()[1]
        n_interactions = np.zeros(len(excel_files), dtype = int)
        for n, filename in enumerate(excel_files):
            for df in excel_files[filename]:

                # List-like interactions
                if df.columns[2] == 'Interaction':
                    only_positives = lambda x: x not in [0, '0', 'ND']
                    n_interactions[n] += df['Interaction'].apply(only_positives).sum()
                # Matrix-like interactions
                else:
                    df = df.set_index(df.columns[0])
                    n_interactions[n] += (df>0).values.sum()

        return n_interactions

    def histogram(self):
        '''
        Plot histogram of the number of positive interactions per file.
        '''

        # Get number of positive interactions per file
        n_interactions = self.interactions_per_file()

        # Plot histogram
        plt.hist(n_interactions, bins = 30)
        plt.xlabel('Number of positive interactions')
        plt.ylabel('Frequency')
        plt.title('Number of positive interactions per file')
        plt.savefig(f'{path.PLOTS}/literaturemining_histogram_interactions_per_file.png')
        plt.clf()

    def interactions_per_taxa(self):
        papers = self.__get_sheets()[0]['Papers']  
        papers['Name'] = papers['Main Author'] + '_' + papers['Year'].astype(str)
        papers['Interactions'] = self.interactions_per_file()
        
        taxa2interactions = defaultdict(int)
        for i, row in papers.iterrows():
            species = row['Species'].split(', ')
            interactions = row['Interactions']
            interactions_per_species = interactions / len(species)
            for specie in species:
                taxa2interactions[specie] += interactions_per_species
        
        # Plot interactors per taxa
        taxa2interactions = sorted(taxa2interactions.items(), key = lambda kv: kv[1], reverse=True)
        plt.bar(*zip(*taxa2interactions))
        plt.xticks(rotation = 90)
        plt.xlabel('Species')
        plt.ylabel('Number of UniProt accessions')
        plt.title('Number of interactors per species')
        plt.tight_layout()
        plt.savefig(f'{path.PLOTS}/literaturemining_interactions_per_species.png')
        plt.clf()

if __name__ == '__main__':
    '''Test class.'''
    literature_mining = LiteratureMining()
    literature_mining.interactions_per_file()
    print(sum(literature_mining.interactions_per_file()))
    literature_mining.histogram()
    literature_mining.interactions_per_taxa()