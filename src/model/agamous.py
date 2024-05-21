
# Third-party modules
import numpy as np
import pandas as pd

# Custom modules
from src.misc import path
from src.tools.esm2 import ESM2
from src.misc.logger import logger
from src.tools.alphafold import AlphaFold
from src.entities.interactor import Interactor

class AGAMOUS:

    def __init__(self):
        arabidopsis = pd.read_excel(f'{path.ARABIDOPSIS}/MIKC.xlsx')
        self.name = arabidopsis['Name'].apply(lambda x: x.split(' ')[0]).values
        self.uniprot_id = arabidopsis['UniProt'].values
        self.interaction = arabidopsis['AG_interaction']
        
    def alphafold(self):
        # AG
        ag_name, ag_uniprot_id = self.name[0], self.uniprot_id[0]
        interactor = Interactor.unpickle(ag_uniprot_id)
        end_kbox = interactor.domains['K-box'][-1][-1]
        start_mads = interactor.domains['MADS-box'][0][0]
        ag_mik_seq = interactor.seq[start_mads:end_kbox]
        
        # All Arabidpsis proteins
        df = {
            #'Name': [], 
            'pLDDT_A_MADS': [], 
            'pLDDT_B_MADS': [], 
            'pLDDT_A_K-box': [], 
            'pLDDT_B_K-box': [], 
            'pLDDT_A_IDOM': [], 
            'pLDDT_B_IDOM': [],
            'pLDDT_A_mean': [],
            'pLDDT_B_mean': [],
            'pLDDT_mean': [],
            'pTM': [],
            'ipTM': [],
            'max_PAE': []
            }
        for name, uniprot_id in zip(self.name, self.uniprot_id):
            
            id = f'{ag_name}_{name}'
            interactor = Interactor.unpickle(uniprot_id)
            kbox = interactor.domains['K-box'][-1]
            mads = interactor.domains['MADS-box'][0]
            mik_seq = interactor.seq[mads[0] : kbox[1]]
            af = AlphaFold(id, f'{ag_mik_seq}:{mik_seq}', f'{path.ALPHAFOLD_AG}/{id}')
            print(name, mik_seq)
            #af.predict()
            metrics = af.metrics()
            plddt_A_mads = np.mean(metrics['rank_1']['pLDDT_A'][mads[0] : mads[1]])
            plddt_B_mads = np.mean(metrics['rank_1']['pLDDT_B'][mads[0] : mads[1]])
            plddt_A_kbox = np.mean(metrics['rank_1']['pLDDT_A'][kbox[0] : kbox[1]])
            plddt_B_kbox = np.mean(metrics['rank_1']['pLDDT_B'][kbox[0] : kbox[1]])
            plddt_A_idom = np.mean(metrics['rank_1']['pLDDT_A'][mads[1] : kbox[0]])
            plddt_B_idom = np.mean(metrics['rank_1']['pLDDT_B'][mads[1] : kbox[0]])

            #df['Name'].append(name)
            df['pLDDT_A_MADS'].append(plddt_A_mads)
            df['pLDDT_B_MADS'].append(plddt_B_mads)
            df['pLDDT_A_K-box'].append(plddt_A_kbox)
            df['pLDDT_B_K-box'].append(plddt_B_kbox)
            df['pLDDT_A_IDOM'].append(plddt_A_idom)
            df['pLDDT_B_IDOM'].append(plddt_B_idom)
            df['pLDDT_A_mean'].append(metrics['rank_1']['pLDDT_A_mean'])
            df['pLDDT_B_mean'].append(metrics['rank_1']['pLDDT_B_mean'])
            df['pLDDT_mean'].append(metrics['rank_1']['pLDDT_mean'])
            df['pTM'].append(metrics['rank_1']['pTM'])
            df['ipTM'].append(metrics['rank_1']['ipTM'])
            df['max_PAE'].append(metrics['rank_1']['max_PAE'])

        df = pd.DataFrame(df)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
        
        return df
    
    def esm2(self):
        data = []

        # AG
        ag_name, ag_uniprot_id = self.name[0], self.uniprot_id[0]
        interactor = Interactor.unpickle(ag_uniprot_id)
        end_kbox = interactor.domains['K-box'][-1][-1]
        start_mads = interactor.domains['MADS-box'][0][0]
        ag_mik_seq = interactor.seq[start_mads:end_kbox]

        # Others
        for name, uniprot_id in zip(self.name, self.uniprot_id):
            
            id = f'{ag_name}_{name}'
            interactor = Interactor.unpickle(uniprot_id)
            kbox = interactor.domains['K-box'][-1]
            mads = interactor.domains['MADS-box'][0]
            mik_seq = interactor.seq[mads[0] : kbox[1]]

            data.append((id, ag_mik_seq + 'G'*25 + mik_seq))

        # ESM2
        esm2 = ESM2()
        d = {}
        for id, seq in data:
            print(id)
            
            esm2.prepare_data([(id, seq)])
            esm2.run_model()
            r, s = esm2.extract_representations()
            d[id] = s[id]

        return pd.DataFrame.from_dict(d, orient='index')

if __name__ == '__main__':
    agamous = AGAMOUS()

    '''
    df = agamous.esm2()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    #from src.machine_learning.dimensionality_reduction import DimensionalityReduction
    #dr = DimensionalityReduction(df, draw = agamous.interaction)
    #dr.pca()
    from lazypredict import LazyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(df.sample(frac=1), agamous.interaction.sample(frac=1), test_size = 0.3)
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    '''




    
    df = agamous.alphafold()
    from src.machine_learning.dimensionality_reduction import DimensionalityReduction
    dr = DimensionalityReduction(df, draw = agamous.interaction, text = [name[-2:] for name in agamous.name])
    dr.pca()
    #X_train, X_test, y_train, y_test = train_test_split(df.sample(frac=1), agamous.interaction.sample(frac=1), test_size = 0.3)
    #clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    #models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    #print(models)
