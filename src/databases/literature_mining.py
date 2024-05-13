# Built-in modules
import os

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path

class LiteratureMining:

    def open_file(self, filename: str) -> pd.DataFrame:
        '''
        Open a file with literature mining data.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        pd.DataFrame
            Dataframe with the literature mining data.
        '''
        filepath = f'{path.LITERATUREMINING}/{filename}'
        return pd.read_excel(filepath)
    
if __name__ == '__main__':
    '''Test class.'''
    n_interactions = []
    literature_mining = LiteratureMining()
    for filename in sorted(os.listdir(path.LITERATUREMINING)):
        print(filename)
        df = pd.read_excel(f'{path.LITERATUREMINING}/{filename}', na_values = ['ND'])
        #print(df)
        print(df)
        if df.columns[2] == 'Interaction':
            n_interactions.append(df['Interaction'].apply(lambda x: x not in [0, '0', 'ND']).sum() )
        else:
            df = df.set_index(df.columns[0])
            n_interactions.append((df>0).values.sum())

        

    print(n_interactions, sum(n_interactions))

#TODO: different sheets in fime