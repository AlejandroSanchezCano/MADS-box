# Built-in modules
import re

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path

class Guidelines:

    def __init__(self):
        # Read the important Excel sheets
        self.papers = pd.read_excel(f'{path.LITERATUREMINING}/Guidelines.xlsx', sheet_name = 'Papers')
        self.species = pd.read_excel(f'{path.LITERATUREMINING}/Guidelines.xlsx', sheet_name = 'Species')

        # Extract the abbreviation from the species column
        find_code = lambda x: re.findall(r'\(([A-Z]{1}[a-z]{1})\)', x)
        self.papers['Abbreviation'] = self.papers['Species'].apply(find_code)
        self.papers = self.papers.explode('Abbreviation')

        # Merge the papers and species dataframes
        self.joined = pd.merge(self.papers, self.species, on = 'Abbreviation')

if __name__ == '__main__':
    guidelines = Guidelines()
    print(guidelines.papers)
    print(guidelines.species)
    print(pd.merge(guidelines.papers, guidelines.species, on = 'Abbreviation'))