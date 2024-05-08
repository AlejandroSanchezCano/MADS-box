# Built-in modules
import requests

# Third-party modules
import bs4
import pandas as pd

# Custom modules
from src.misc import path
from src.misc import utils
from src.misc.logger import logger

class PlaPPISite:

    # Override static attributes
    database = 'PlaPPISite'

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
    
    def map_ids(self) -> None:
        '''
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

        # How many are mapped? (1)
        n_mapped_ids = 0

        # Retrieve PPI table of all MADS proteins   
        for interactor in utils.iterate_folder(path.INTERACTORS):
            soup = self._soupify(interactor.uniprot_id)
            table = self._get_table(soup)
            non_predicted_table = table[table['PPI source'].apply(lambda x: x not in ['Predicted', 'prediction'])]
            
            # Mapped ID when UniProt accession has experimentally verified PPIs
            if not non_predicted_table.empty:
                
                # Update Interactor method
                interactor.plappisite_id = interactor.uniprot_id
                interactor.pickle()

                # How many are mapped? (2)
                n_mapped_ids += 1
                logger.info(f'{interactor.uniprot_id} has PPIs in {self.database} ({n_mapped_ids}th mapped ID)')

        # Logging
        logger.info(f'{n_mapped_ids} MIKC UniProt IDs have PPIs in {self.database}')        

if __name__ == '__main__':
    '''Test class'''
    plappisite = PlaPPISite()
    plappisite.map_ids()
