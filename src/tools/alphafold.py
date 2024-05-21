# Built-in modules
import os
import json
import subprocess

# Third-party modules
import numpy as np

# Custom modules
from src.misc.logger import logger

class AlphaFold:

    def __init__(self, id: str, seq: str, output_dir: str):
        self.id = id
        self.seq = seq
        self.output_dir = output_dir

    def predict(
            self, 
            # Relaxation
            amber: bool = True,
            num_relax: int = 1,
            # Prediction
            num_recycle: int = 4,
            num_ensemble: int = 2,
            random_seed: int = 573,
            num_models: int = 5,
            ) -> dict[str, dict[str, float|np.ndarray]]:
        '''
        _summary_

        Parameters
        ----------
        id : str
            Protein (complex) ID
        seq : str
            Protein (complex) sequence
        output_dir : str
            Output directory
        amber : bool, optional
            Enable OpenMM/Amber for structure relaxation. Can improve
            the quality of side-chains at a cost of longer runtime. By
            default True
        num_relax : int, optional
            Specify how many of the top ranked structures to relax using 
            OpenMM/Amber. Typically, relaxing the top-ranked prediction 
            is enough and speeds up the runtime, so by default 1
        num_recycle : int, optional
            Number of prediction recycles. Increasing recycles can 
            improve the prediction quality but slows down the 
            prediction., by default 4
        num_ensemble : int, optional
            Number of ensembles. The trunk of the network is run 
            multiple times with different random choices for the MSA 
            cluster centers. This can result in a better prediction at 
            the cost of longer runtime, by default 2
        random_seed : int, optional
            Changing the seed for the random number generator can result 
            in better/different structure predictions, by default 573
        num_models : int, optional
            Number of models to use for structure prediction. Reducing 
            the number of models speeds up the prediction but results in 
            lower quality, by default 5

        Currently, amber relaxation not supported because openmm needs 
        GLIBCXX_3.4.26, which included in versions gxx >= 9.0

        Returns
        -------
        dict[str, dict[str, float|np.ndarray]]
            Result metrics as a dictionary of dictionaries:
            {'rank1': {'pLDDT': np.ndarray, 'pLDDT_mean': float,
                        'pLDDT_A': float, 'pLDDT_B': float,
                        'pTM': float, 'ipTM': float,
                        'max_PAE': float, 'PAE': np.ndarray}}
        '''
        # Make output directory
        os.makedirs(self.output_dir, exist_ok = True)

        # FASTA file
        with open(f'{self.output_dir}/{id}.fa', 'w') as f:
            f.write(f'>{id}\n{self.seq}')

        # Run AlphaFold
        cmd = [
            'colabfold_batch',
            f'{self.output_dir}/{id}.fa',
            f'{self.output_dir}',
            # '--amber' if amber else '',
            # f'--num-relax {num_relax}',
            f'--num-recycle {num_recycle}',
            f'--num-ensemble {num_ensemble}',
            f'--random-seed {random_seed}',
            f'--num-models {num_models}'
        ]
        subprocess.run(' '.join(cmd), shell = True)

        # Logger
        logger.info(f'AlphaFold prediction for {id}')

    def metrics(self) -> dict[str, dict[str, float|np.ndarray]]:
        # Process metric output
        scores = sorted([json for json in os.listdir(self.output_dir) if json.endswith('.json') and json.startswith(self.id + '_scores')])
        metrics = {f'rank_{n + 1}':{} for n in range(len(scores))}
        # Iterate over score file names
        for score, metric in zip(scores, metrics):
            with open(f'{self.output_dir}/{score}', 'r') as f:
                # Open json file
                data = json.load(f)
                # Extract metrics
                metrics[metric]['pLDDT'] = data['plddt']
                metrics[metric]['pLDDT_mean'] = np.mean(data['plddt'])
                metrics[metric]['pLDDT_A'] = data['plddt'][:len(self.seq.split(':')[0])]
                metrics[metric]['pLDDT_B'] = data['plddt'][len(self.seq.split(':')[0]):]
                metrics[metric]['pLDDT_A_mean'] = np.mean(data['plddt'][:len(self.seq.split(':')[0])])
                metrics[metric]['pLDDT_B_mean'] = np.mean(data['plddt'][len(self.seq.split(':')[0]):])
                metrics[metric]['pTM'] = data['ptm']
                metrics[metric]['ipTM'] = data['iptm']
                metrics[metric]['max_PAE'] = data['max_pae']
                #metrics[metric]['PAE'] = data['pae']
        
        return metrics

if __name__ == '__main__':
    af = AlphaFold()
    af.predict(
        id = 'test', 
        seq = 'MKKFFLIGLVLLFSSVSAA:MKKFFLIGLVLLFSSVSAA',
        output_dir = 'output'
    )