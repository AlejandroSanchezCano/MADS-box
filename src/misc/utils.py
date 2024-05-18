# Built-in modules
from typing import Any, Tuple

# Third-party modules
import pickle

def pickling(data: Any, path: str) -> None:
    '''
    Pickle an object and store it.

    Parameters
    ----------
    data : Any
        Pickable object that will be stored.
    path : str
        Storing path.
    '''
    with open(path, 'wb') as handle:
        pickle.dump(
            obj = data,
            file = handle, 
            protocol = pickle.HIGHEST_PROTOCOL
            )

def unpickling(path: str) -> Any:
    '''
    Retrieves and unpickles a pickled object.

    Parameters
    ----------
    path : str
        Storing path of the object to unpickle.

    Returns
    -------
    Any
        Unpickled object.
    '''
    with open(path, 'rb') as handle:
        return pickle.load(file = handle)

def read_fasta_str(fasta: str) -> Tuple[str, str]:
    '''
    Parses a FASTA string into its header and sequence.

    Parameters
    ----------
    fasta : str
        FASTA string.

    Returns
    -------
    tuple[str, str]
        Tuple containing the header and the sequence.
    '''
    header, sequence = fasta.split('\n', 1)
    header = header.replace('>', '')
    sequence = sequence.replace('\n', '')

    return header, sequence