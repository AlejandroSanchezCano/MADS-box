# Built-in libraries
import time
from functools import wraps
from typing import Callable
from string import ascii_uppercase, ascii_lowercase

# Third-party libraries
import torch
import numpy as np
import transformers
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

# Custom libraries
from src.misc.logger import logger

# TODO: substitute A, B, C ticks in PAE with protein names

class ESMFold():
    # huggingface conda environment
    # pip install --upgrade transformers py3Dmol accelerate
    # pip install --upgrade nvitop
    # mamba install scipy
    # mamba install -c conda-forge matplotlib
    # mamba install seaborn -c conda-forge
    # pip install --upgrade biopython
    
    def __init__(self):
        self.tokenizer, self.model = self.load()
        self.seqs = None
        self.output = None

    def load(self) -> tuple[
        transformers.models.esm.tokenization_esm.EsmTokenizer,
        transformers.models.esm.modeling_esmfold.EsmForProteinFolding
        ]:
        '''
        Loads tokenizer and ESMFold model

        Returns
        -------
        tuple[ transformers.models.esm.tokenization_esm.EsmTokenizer, transformers.models.esm.modeling_esmfold.EsmForProteinFolding ]
            Tokenizer, ESMFold model
        '''
        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = transformers.EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

        # Logger
        logger.info('Tokenizer and model are loaded')

        return tokenizer, model

    def performance_optimizations(self) -> None:
        '''
        Optimizes model and paramaters for GPU computations and less memory.
        '''
        # Move model to GPU
        self.model = self.model.cuda()

        # Convert model stem to float16
        self.model.esm = self.model.esm.half()
        
        # Enable TensorFloat32 computation
        torch.backends.cuda.matmul.allow_tf32 = True

        # Reduce 'chunk size' because our GPU memory > 16GB and len(sequences) < 600
        self.model.trunk.set_chunk_size(64)

        # Logger
        logger.info('Model is optimized')

    def timer(function: Callable) -> Callable: 
        '''
        Measures the time taken to fold a sequence.

        Parameters
        ----------
        function : Callable
            fold function

        Returns
        -------
        Callable
            Wrapper function
        '''
        @wraps(function)
        def wrapper(*args, **kwargs): 
            t1 = time.time() 
            result = function(*args, **kwargs) 
            t2 = time.time() 
            logger.info(f'Sequences folded in {(t2-t1):.4f}s') 
            return result 
        return wrapper 
    
    @timer
    def fold(
            self, 
            seqs: list[str],
            linker: str = 'G'*25, 
            residue_index_offset: int = 512,
            ) -> transformers.models.esm.modeling_esmfold.EsmForProteinFoldingOutput:
        '''
        Uses ESMFold to fold a protein sequence.

        Parameters
        ----------
        seqs : list[str]
            Sequences to fold
        linker : str, optional
            Trick to fold multiple chains from the paper, by default 'G'*25, a
            linker of flexible glycine residues
        residue_index_offset : int, optional
            Offset the position IDs of the different chains by default 512, so
            that the model treats them as distant portions of the same long 
            chain, not one single peptide.

        Returns
        -------
        transformers.models.esm.modeling_esmfold.EsmForProteinFoldingOutput
            ESMFold output.
        '''

        # Create peptide
        self.seqs = seqs
        protein = linker.join(seqs)

        # Tokenizer -> {'input_ids':[], 'attention_mask':[]}
        tokenized = self.tokenizer([protein], return_tensors="pt", add_special_tokens=False)

        # Offset positions of chains
        with torch.no_grad():
            position_ids = torch.arange(len(protein), dtype=torch.long)
            cumulative_length = np.cumsum([len(seq) for seq in seqs])
            for n, cum_len in enumerate(cumulative_length[:-1]):
                position_ids[cum_len + n*len(linker):] += residue_index_offset

        # Add offset positions to the tokenized inputs
        tokenized['position_ids'] = position_ids.unsqueeze(0)

        # Move to GPU
        tokenized = {key: tensor.cuda() for key, tensor in tokenized.items()}

        # Fold
        with torch.no_grad():
            output = self.model(**tokenized)

        # Mask linker
        binary_masks = [[1]*len(seq) + [0]*len(linker) for seq in seqs[:-1]] + [[1]*len(seqs[-1])]
        flattened_mask = [mask for masks in binary_masks for mask in masks]
        mask = torch.tensor(flattened_mask)[None, :, None]
        output['atom37_atom_exists'] *= mask.to(output['atom37_atom_exists'].device)

        self.output = output
        return output

    def plot_plddt(self) -> None:
        '''Plots pLDDT values.'''
        # Mask linker
        plddt = self.output["plddt"].cpu().numpy()[0,:,1]
        mask = self.output["atom37_atom_exists"].cpu().numpy()[0,:,1] == 1

        # Scatterplot and rugplot
        y = plddt[mask] * 100
        x = list(range(len(y)))
        hue = ['Very high' if i >= 90 else 'Condifent' if i >= 70 else 'Low' if i >= 50 else 'Very low' for i in y]
        plddt_palette = {'Very high': '#0053D6', 'Condifent':'#65CBF3', 'Low':'#FFDB13', 'Very low':'#FF7D45'}
        fig = sns.scatterplot(x=x, y=y, color='black')
        sns.rugplot(x=x, hue = hue, palette = plddt_palette, legend = False, linewidth = 2, expand_margins=True)

        # Horizontal spans
        ax = plt.gca()
        ax.axhspan(90, 100, alpha = 0.5, color = '#0053D6')
        ax.axhspan(70, 90, alpha = 0.5, color = '#65CBF3')
        ax.axhspan(50, 70, alpha = 0.5, color = '#FFDB13')
        ax.axhspan(0, 50, alpha = 0.5, color = '#FF7D45')
        ax.set_ylim(0, 100)

        # Vertical sepratory lines
        lengths = [len(seq) for seq in self.seqs]
        chain_separation = np.cumsum(lengths)
        plt.vlines(x = chain_separation[:-1], ymin = 0, ymax = chain_separation[-1], colors = 'black')

        # Chain ticks
        ticks = np.append(0, chain_separation)
        ticks = (ticks[1:] + ticks[:-1])/2
        alphabet_list = list(ascii_uppercase+ascii_lowercase)
        plt.xticks(ticks, alphabet_list[:len(ticks)])

        # Axis labels
        fig.set_xlabel('Residue index')
        fig.set_ylabel('pLDDT')

        # Save plot
        plt.savefig(f'plddt_pi_svp.png', bbox_inches='tight')
        plt.show()
    
    def plot_extra(self) -> None:
        '''
        Plots horizontal and vertical lines separating chains in symmetric
        matrices as well as the name ticks for each resulting box.
        '''
        # Horizontal and vertical lines
        lengths = [len(seq) for seq in self.seqs]
        chain_separation = np.cumsum(lengths)
        plt.vlines(x = chain_separation[:-1], ymin = 0, ymax = chain_separation[-1], colors = 'black')
        plt.hlines(y = chain_separation[:-1], xmin = 0, xmax = chain_separation[-1], colors = 'black')

        # Ticks
        ticks = np.append(0, chain_separation)
        ticks = (ticks[1:] + ticks[:-1])/2
        alphabet_list = list(ascii_uppercase+ascii_lowercase)
        plt.xticks(ticks, alphabet_list[:len(ticks)])
        plt.yticks(ticks, alphabet_list[:len(ticks)])


    def plot_pae(self) -> None:
        '''Plots predicted aligned error (PAE)'''
        # Mask linker
        pae_all = (self.output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(-1) * 31
        mask = self.output["atom37_atom_exists"].cpu().numpy()[0,:,1] == 1
        pae_masked = pae_all[mask,:][:,mask]
        
        # Plot
        plt.figure()
        plt.title('Predicted Aligned Error')
        plt.imshow(pae_masked,cmap="bwr",vmin=0,vmax=30,extent=(0, pae_masked.shape[0], pae_masked.shape[0], 0))
        self.plot_extra()
        cb = plt.colorbar()
        plt.xlabel('Scored residue')
        plt.ylabel('Aligned residue')
        plt.savefig(f'pae_pi_svp.png', bbox_inches='tight')
        cb.remove()

    def plot_contact_map(self) -> None:
        '''Plots map of contacts between residues'''
        # Mask linker
        bins = np.append(0,np.linspace(2.3125,21.6875,63))
        sm_contacts = softmax(self.output["distogram_logits"].cpu().numpy(),-1)[0]
        sm_contacts = sm_contacts[...,bins<8].sum(-1)
        mask = self.output["atom37_atom_exists"].cpu().numpy()[0,:,1] == 1
        sm_contacts_masked = sm_contacts[mask,:][:,mask]
        n_residues = sm_contacts_masked.shape[0]

        # Plot
        plt.figure()
        plt.title("Contact map")
        plt.xlabel('Residue position')    
        plt.ylabel('Residue position')  
        plt.imshow(sm_contacts_masked, cmap="Greys", vmin=0, vmax=1, extent=(0, n_residues, n_residues, 0))
        self.plot_extra()
        plt.savefig(f'contact_map_pi_svp.png', bbox_inches='tight')

    def save_pdb(self, id: str) -> None: 
        pdb = self.model.output_to_pdb(self.output)[0]   
        with open(f"pdb_pi_sep3.pdb","w") as handle:
            handle.write(pdb)

if __name__ == '__main__':
    sep3 = 'MGRGRVELKRIENKINRQVTFAKRRNGLLKKAYELSVLCDAEVALIIFSNRGKLYEFCSSSSMLRTLERYQKCNYGAPEPNVPSREALAVELSSQQEYLKLKERYDALQRTQRNLLGEDLGPLSTKELESLERQLDSSLKQIRALRTQFMLDQLNDLQSKERMLTETNKTLRLRLADGYQMPLQLNPNQEEVDHYGRHHHQQQQHSQAFFQPLECEPILQIGYQGQQDGMGAGPSVNNYMLGWLPYDTNSI'
    pi = 'MGRGKIEIKRIENANNRVVTFSKRRNGLVKKAKEITVLCDAKVALIIFASNGKMIDYCCPSMDLGAMLDQYQKLSGKKLWDAKHENLSNEIDRIKKENDSLQLELRHLKGEDIQSLNLKNLMAVEHAIEHGLDKVRDHQMEILISKRRNEKMMAEEQRQLTFQLQQQEMAIASNARGMMMRDHDGQFGYRVQPIQPNLQEKIMSLVID'
    svp = 'MAREKIQIRKIDNATARQVTFSKRRRGLFKKAEELSVLCDADVALIIFSSTGKLFEFCSSSMKEVLERHNLQSKNLEKLDQPSLELQLVENSDHARMSKEIADKSHRLRQMRGEELQGLDIEELQQLEKALETGLTRVIETKSDKIMSEISELQKKGMQLMDENKRLRQQGTQLTEENERLGMQICNNVHAHGGAESENAAVYEEGQSSESITNAGNSTGAPVDSESSDTSLRLGLPYGG'
    
    esm = ESMFold()
    esm.performance_optimizations()
    output = esm.fold([sep3])
    esm.plot_plddt()
    esm.plot_pae()
    esm.plot_contact_map()
    #esm.save_pdb('sadasd')