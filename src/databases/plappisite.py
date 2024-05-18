# Built-in modules
import requests

# Third-party modules
import bs4
import pandas as pd

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.entities.interactor import Interactor

class PlaPPISite:

    def _soupify(self, uniprot_id: str) -> bs4.BeautifulSoup:
        '''
        PlaPPISite has no API and their downloadable files are not 
        suitable for mapping IDs, so the contents need to be retrieved 
        by parsing the web content. Retrieves the web content of a 
        specific UniProt ID as a BeautifulSoup object.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID to search in PlaPPISite.

        Returns
        -------
        bs4.BeautifulSoup
            Soupified web content.
        '''
        url = f'http://zzdlab.com/plappisite/single_idmap.php?protein={uniprot_id}'
        web = requests.get(url).text
        soup = bs4.BeautifulSoup(web, features= 'lxml')
        return soup

    def _get_table(self, soup: bs4.BeautifulSoup) -> pd.DataFrame:
        '''
        Parses the web content of an accession and retrieves the PPI 
        table.

        Parameters
        ----------
        soup : bs4.BeautifulSoup
            Accession's web content.

        Returns
        -------
        pd.DataFrame
            Interaction table.
        '''
        table = soup.find('div', attrs = {'id':'container_table'})
        columns = [th.text for th in table.find_all('th')]
        tds = [td.text for td in table.find_all('td')]
        rows = [tds[i : i + len(columns)] for i in range(0, len(tds), len(columns))]

        return pd.DataFrame(rows, columns = columns)
    
    def mads_vs_all(self) -> None:
        '''
        Searches for MADS interactors in PlaPPISite and retrieves their
        PPIs.
        PlaPPISite uses UniProt IDs as main IDs, so each UniProt IDs is 
        checked to have PPIs in PlaPPISite by parsing the web content, 
        retrieving the PPI table and removing the predicted PPIs. If the
        resulting PPI table is not empty, the UniProt ID is considered
        to have PPIs in PlaPPISite and the Interactor.plappisite_id 
        attribute is updated. 
        The predicted PPIs are removed because STRING is arguably the
        best source of predicted PPIs, so PlaPPISite predicted PPIs will 
        not be likely used.
        '''
        # Retrieve PPI table of all MADS proteins   
        for interactor in Interactor.iterate_folder(path.INTERACTORS):
            soup = self._soupify(interactor.uniprot_id)
            table = self._get_table(soup)
            non_predicted_table = table[table['PPI source'].apply(lambda x: x not in ['Predicted', 'prediction'])]
            
            # Append to DataFrame if not empty (non-predicted PPIs)
            if not non_predicted_table.empty:
                
                # Load DataFrame or create it if there is none in cache
                try:
                    cache_path = f'{path.NETWORKS}/PlaPPISite_MADS_vs_ALL.tsv'
                    mads_vs_all = pd.read_csv(cache_path, sep = '\t')
                except FileNotFoundError:
                    mads_vs_all = pd.DataFrame() 
                
                # Append to DataFrame
                mads_vs_all = pd.concat([mads_vs_all, non_predicted_table], ignore_index = True)
                mads_vs_all.to_csv(cache_path, sep = '\t', index = False)

        # Logging
        logger.info(f'MADS vs. all PPIs in PlaPPISite -> dim({mads_vs_all.shape})')

    def mads_vs_mads(self) -> None:
        '''
        Filters MADS vs. MADS interactions from the MADS vs. ALL
        '''
        # Load MADS_vs_ALL DataFrame
        filepath = f'{path.NETWORKS}/PlaPPISIte_MADS_vs_ALL.tsv'
        mads_vs_all = pd.read_csv(filepath, sep = '\t')

        # MADS UniProt IDs
        mads = set([interactor.uniprot_id for interactor in Interactor.iterate_folder(path.INTERACTORS)])

        # Filter MADS vs. ALL DataFrame
        is_there_mikc = lambda x: set(x.split(' - ')).issubset(mads)
        mads_vs_mads = mads_vs_all[mads_vs_all['PPI'].apply(is_there_mikc)]

        # Save DataFrame
        filepath = f'{path.NETWORKS}/PlaPPISite_MADS_vs_MADS.tsv'
        mads_vs_mads.to_csv(filepath, sep = '\t', index = False)

        # Loggi

if __name__ == '__main__':
    '''Test class'''
    plappisite = PlaPPISite()
    plappisite.mads_vs_all()
    # plappisite.mads_vs_mads()
