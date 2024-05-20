
# Third-party modules
import pandas as pd

# Custom modules
from src.misc import path
from src.misc.logger import logger
from src.tools.alphafold import AlphaFold
from src.entities.interactor import Interactor

class AGAMOUS:

    def __init__(self):
        arabidopsis = pd.read_excel(f'{path.ARABIDOPSIS}/MIKC.xlsx')
        self.name = arabidopsis['Name'].apply(lambda x: x.split(' ')[0]).values
        self.uniprot_id = arabidopsis['UniProt'].values
        
    def alphafold(self):
        # AG
        ag_name, ag_uniprot_id = self.name[0], self.uniprot_id[0]
        interactor = Interactor.unpickle(ag_uniprot_id)
        end_kbox = interactor.domains['K-box'][-1][-1]
        start_mads = interactor.domains['MADS-box'][0][0]
        ag_mik_seq = interactor.seq[start_mads:end_kbox]
        
        # All Arabidpsis proteins
        for name, uniprot_id in zip(self.name, self.uniprot_id):
            id = f'{ag_name}_{name}'
            interactor = Interactor.unpickle(uniprot_id)
            end_kbox = interactor.domains['K-box'][-1][-1]
            start_mads = interactor.domains['MADS-box'][0][0]
            mik_seq = interactor.seq[start_mads:end_kbox]
            AlphaFold().predict(id, f'{ag_mik_seq}:{mik_seq}', f'{path.ALPHAFOLD_AG}/{id}')

if __name__ == '__main__':
    agamous = AGAMOUS()
    agamous.alphafold()