# Built-in libraries
from typing import Literal

# Third-party libraries
import esm
import torch
import matplotlib.pyplot as plt

# Custom libraries
from src.misc.logger import logger

class ESM2:

    # huggingface conda environment
    # pip install --upgrade transformers py3Dmol accelerate
    # pip install --upgrade nvitop
    # mamba install scipy
    # mamba install -c conda-forge matplotlib
    # mamba install seaborn -c conda-forge
    # pip install sentencepiece  
    # pip install protobuf
    # pip install fair-esm  
    # mamba install -c bioconda multitax
    # pip install bioservices
    # mamba install -c conda-forge scikit-learn
    # pip install umap-learn

    def __init__(self):
        self.model, self.alphabet = self.load()
        self.batch = None
        self.output = None

    def load(self) -> tuple[esm.model.esm2.ESM2, esm.data.Alphabet]:
        '''
        Load ESM2 model and alphabet.

        Returns
        -------
        tuple[esm.model.esm2.ESM2, esm.data.Alphabet]
            ESM2 model and alphabet.
        '''

        # Load model and alphabet
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        # Disables dropout for deterministic results
        model.eval()

        # GPU/CPU
        model = model.cuda()

        # Logger
        logger.info('Model is loaded')

        return model, alphabet
    
    def prepare_data(self, data: list[tuple[str, str]]) -> None:
        '''
        Prepare data for embedding using alphabet.get_batch_converter() method.

        Parameters
        ----------
        data : list[tuple[str, str]]
            List of tuples containing the protein label (id) and the protein 
            sequence. It is recommended to process one protein at a time to 
            avoid memory issues.
        '''
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        
        self.batch = batch_labels, batch_strs, batch_tokens, batch_lens
        
    def run_model(self) -> dict[Literal['logits', 'representations', 'attentions', 'contacts'], torch.Tensor]:
        '''
        Run ESM2.

        Returns
        -------
        dict[Literal['logits', 'representations', 'attentions', 'contacts'], torch.Tensor]
            Dictionary containing the logits, representations, attentions, and
            contacts results of the model.
        '''

        # Data
        _, _, batch_tokens, _ = self.batch
        
        # Run model
        with torch.no_grad():
            output = self.model(batch_tokens.cuda(), repr_layers=[33], return_contacts = True)
        
        # Logging
        logger.info('Embeddings are generated')

        self.output = output
        return output

    def extract_representations(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        '''
        Extract per-residue and per-sequence representations/embeddings.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            Tuple containing the per-residue and per-sequence representations.
        '''
        # Data
        labels, _, _, lens = self.batch
        
        # Output
        representations = self.output["representations"][33]

        # Per-residue and per-sequence representations
        per_residue = {label:embedding.cpu().numpy() for label, embedding in zip(labels, representations)}
        per_sequence = {label:representations[i, 1 : tokens_len - 1].mean(0).cpu().numpy() for i, (label, tokens_len) in enumerate(zip(labels, lens))}

        # Logger
        logger.info('Representations are extracted')

        return per_residue, per_sequence
    
    def contact_maps(self, save_folder: str) -> None:
        '''
        Extract contact maps from the model output and save them as pngs.

        Parameters
        ----------
        save_folder : str
            Folder where the contact maps pngs will be saved.
        '''
        # Data
        labels, _, _, lens = self.batch

        # Plot
        for label, tokens_len, attention_contacts in zip(labels, lens, self.output["contacts"].cpu().numpy()):
            plt.matshow(attention_contacts[: tokens_len, : tokens_len])
            plt.title(f'{label} contact map')
            plt.savefig(f"{save_folder}/{label}.png", transparent = True)
            plt.close()

        # Logger
        logger.info('Contact maps are extracted')

if __name__ == '__main__':
    sep3 = 'SGRGKIEIKRIENTTNRQVTFCKRRNGLLKKAYELSVLCDAEVALIVFSSRGRLYEYSNNSVKGTIERYKKAISDNSNTGSVAEINAQYYQQESAKLRQQIISIQNSNRQLMGETIGSMSPKELRNLEGRLERSITRIRSKKNELLFSEIDYMQKREVDLHNDNQILRAKIAENERN'
    ag = 'MGRGRVELKRIENKINRQVTFAKRRNGLLKKAYELSVLCDAEVALIIFSNRGKLYEFCSSSSMLRTLERYQKCNYGAPEPNVPSREALAVELSSQQEYLKLKERYDALQRTQRNLLGEDLGPLSTKELESLERQLDSSLKQIRALRTQFMLDQLNDLQSKERMLTETNKTLRLRLADGYQM'
    data = [
        ("ag_sep3", ag + 'G'*25 + sep3),
        ('ag', ag)
        ]
    esm2 = ESM2()
    esm2.prepare_data(data)
    esm2.run_model()
    r, s = esm2.extract_representations()
    esm2.contact_maps('.')
    print([len(i) for i in list(s.values())])