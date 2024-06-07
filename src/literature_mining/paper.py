# Built-in modules
import os
import re
from typing import Generator, Any

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from src.misc import path

class Paper:

    def __init__(self, author: str, year: str):
        self.author = author
        self.year = year
        self.dfs = list(pd.read_excel(
            f'{path.LITERATUREMINING}/Excels/{author}_{year}.xlsx', 
            sheet_name = None,
            na_values = 'ND',
            header = 0
            ).values())
    '''
    @property
    def species(self) -> list:
        joined = Guidelines().joined
        info = joined[joined['Main Author'] == self.author]
        return info['NCBI'].values
    '''

    @property
    def name2uniprot(self) -> dict:
        '''
        Load the dictionary that maps gene names to UniProt IDs.

        Returns
        -------
        dict
            Gene name to UniProt ID mapping.
        '''
        dictionary = pd.read_excel(f'{path.LITERATUREMINING}/DICT.xlsx')
        return dictionary.set_index('Name').to_dict()['UniProt ID']
            
    def __repr__(self):
        return f'{self.author} ({self.year})'

    def matrix_or_list(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Determine if the DataFrame is a matrix or a list and process it.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be analyzed.

        Returns
        -------
        pd.DataFrame
            Processed sheet.
        '''
        if df.columns[2] == 'Interaction':
            return self.process_list(df)
        else:
            return self.process_matrix(df)
    
    def parse_gene_name(self, name: str) -> list[str, str, str]: 
        '''
        Parse the gene name into its components: species, gene, and mutation.

        Parameters
        ----------
        name : str
            Gene name to be parsed.

        Returns
        -------
        list[str, str, str]
            List of strings with the species, gene, and mutation components.
            
        '''
        # Split the name into components: species, gene, mutations
        components = ['', '', '']

        # Species component
        components[0] = name[:2]

        # Mutation component
        mutation = name.split('_')[1:] if '_' in name else ''
        components[2] = '_'.join(mutation)
        
        # Gene component
        gene = name.split('_')[0][2:]   
        components[1] = gene

        return components
    
    def process_list(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process a list DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame.
        '''
        df = df.rename(columns = {df.columns[0] : 'A', df.columns[1] : 'B'})
        to_uniprot = lambda x: self.name2uniprot[''.join(self.parse_gene_name(x)[:2])] + ('_' + self.parse_gene_name(x)[2] if self.parse_gene_name(x)[2] else '')
        df['A'] = df['A'].apply(to_uniprot)
        df['B'] = df['B'].apply(to_uniprot)
        df['A-B'] = df.apply(lambda x: '-'.join(sorted([x['A'], x['B']])), axis = 1)
        df = df[~df['A'].str.contains('NONE')]
        df = df[~df['B'].str.contains('NONE')]
        return df.dropna()
    
    def process_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process a matrix DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame.
        '''
        # First column as index
        df = df.set_index(df.columns[0])

        # Column names to UniProt IDs + mutation
        for column in df.columns:
            species, gene, mutation = self.parse_gene_name(column)
            mutation = '_' + mutation if mutation else ''
            df = df.rename(columns = {column : f'{self.name2uniprot[species + gene]}{mutation}'})

        # Index names to UniProt IDs + mutation
        for index in df.index:
            species, gene, mutation = self.parse_gene_name(index)
            mutation = '_' + mutation if mutation else ''
            df = df.rename(index = {index : f'{self.name2uniprot[species + gene]}{mutation}'})
        
        # Remove NONE columns and indexes
        df = df.loc[:, ~df.columns.str.contains('NONE')]
        df = df.loc[~df.index.str.contains('NONE')]

        # Convert to list-like DataFrame
        dff = {'A':[], 'B':[], 'Interaction':[], 'A-B':[]}
        for i in df.columns:
            for j in df.index:
                dff['A'].append(i)
                dff['B'].append(j)
                dff['Interaction'].append(df.loc[j][i])
                dff['A-B'].append('-'.join(sorted([i, j])))

        dff = pd.DataFrame(dff).sort_values('A-B')

        # Drop NaNs
        dff = dff.dropna()
        
        # Remove duplicates
        to_remove = []
        for interaction in dff['A-B'].unique():
            int_df = dff[dff['A-B'] == interaction]['Interaction']
            index = int_df.index.values
            interaction = int_df.values
            if len(index) == 2:
                if bool(int(interaction[0])) != bool(int(interaction[1])):
                    to_remove.append(index[0] if not interaction[0] else index[1])
                else:
                    to_remove.append(index[0])
        dfff = dff.drop(to_remove)

        return dfff

    @staticmethod
    def iterate() -> Generator[Any, None, None]:
        '''
        Iterate over the literature mining papers (except Guidelines).

        Yields
        ------
        Generator[Any, None, None]
            Paper object.
        '''
        for paper in tqdm(sorted(os.listdir(f'{path.LITERATUREMINING}/Excels'))):
                name, year, _ = re.split(r'[_\.]', paper)
                yield Paper(name, year)

    @staticmethod
    def analyse_papers() -> pd.DataFrame:
        '''
        Analyze all the literature mining papers.

        Returns
        -------
        pd.DataFrame
            Database with all the interactions.
        '''
        db = pd.DataFrame()
        for paper in Paper.iterate():
            for df in paper.dfs:
                db = pd.concat([db, paper.matrix_or_list(df)])
        return db.drop_duplicates('A-B')
    
if __name__ == '__main__':
    p = Paper('VanDijk', '2010')
    for df in p.dfs:
        print(p.matrix_or_list(df))