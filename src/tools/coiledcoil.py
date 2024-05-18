# Built-in modules

# Third-party modules
import deepcoil
import numpy as np

# Custom modules
from src.misc.logger import logger

class DeepCoil:

    def __init__(self):
        # It doesn't find GPU but it is actually faster with GPU
        self.model = deepcoil.DeepCoil(use_gpu = False) 
        self.prediction = None

    def predict(self, id: str, seq: str) -> 'dict[str, np.ndarray]':
        '''
        Predicts coiled-coil regions in a protein sequence using 
        DeepCoil.

        Parameters
        ----------
        id : str
            Protein ID
        seq : str
            Protein sequence

        Returns
        -------
        dict[str, np.ndarray]
            DeepCoil predictions {'cc': np.ndarray, 'hept': np.ndarray}
        '''
        # Run model
        fasta_dict = {id : seq}
        self.prediction = self.model.predict(fasta_dict)[id]

        # Logger
        logger.info(f'Coiled-coil prediction for {id} is done')

        return self.prediction

    def a(self, threshold: float = 0.2) -> 'list[int]':
        '''
        Returns the indices of the amino acids that are predicted to be
        the 'a' positions in the coiled-coil heptad.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability threshold, by default 0.2

        Returns
        -------
        list[int]
            List of a indices.
        '''
        return np.argwhere(self.prediction['hept'][:, 1] > threshold).flatten()
    
    def d(self, threshold: float = 0.2) -> 'list[int]':
        '''
        Returns the indices of the amino acids that are predicted to be
        the 'd' positions in the coiled-coil heptad.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability threshold, by default 0.2

        Returns
        -------
        list[int]
            List of d indices.
        '''
        return np.argwhere(self.prediction['hept'][:, 2] > threshold).flatten()
    
    def coiledcoil(self) -> 'list[tuple[int, int]]':
        '''
        Returns the start and end indices of the coiled-coil regions by
        detecting the peaks in the coiled-coil prediction using the
        'sharpen_preds' function from the deepcoil 'utils' module.

        Returns
        -------
        list[tuple[int, int]]
            List of tuples containing the start and end indices of the
            coiled-coil regions.
        '''
        peaks = deepcoil.utils.sharpen_preds(self.prediction['cc'])
        diff = np.diff(peaks)
        start = np.argwhere(diff > 0)[:, 0] + 1
        end = np.argwhere(diff < 0)[:, 0]

        return list(zip(start, end))
    
    def plot(self) -> None:
        '''
        Plots the coiled-coil prediction using the 'plot_preds' function
        from the deepcoil 'utils' module.
        '''
        deepcoil.utils.plot_preds(
            result = self.prediction, 
            beg = 0,
            end = -1,
            out_file = None,
            dpi = 300
            )
    
if __name__ == '__main__':
    deepcoil = DeepCoil()
    r = deepcoil.predict('Q9UJU2', 'MEDQFSIYFSTLNTLPSKPNPTHSFFFFHFSSLKSNLPFFFSSFTSSHQRFVNFSSYPIFKISTTEFPNQSGEGSASSQKKMGRGKIEIKRIENTTNRQVTFCKRRNGLLKKAYELSVLCDAEVALIVFSTRGRLYEYANNSVRGTIERYKKAFADSSNSGLSVAEANVQFYQQEATKLKRQIREIQNSNRHILGEALSSLPLKELKSLEGRLERGISKVRAKKNETLFAEMEFMQKREMELQSHNNYLRTQIAEHERIQQQQQQQQQTNMMQRATYESVGGQYDDENRSTYGAVGALMDSDSHYAPQDHLTALQLV')
    print(deepcoil.coiledcoil())