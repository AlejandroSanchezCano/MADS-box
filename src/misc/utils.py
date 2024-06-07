# Built-in modules
from typing import Any, Generator, Tuple

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

def read_fasta_str(file: str) -> Generator[Tuple[str, str], None, None]:
    '''
    Parses a FASTA file into its headers and sequences.

    Parameters
    ----------
    file : str
        FASTA file.

    Returns
    -------
    Generator[Tuple[str, str], None, None]
        Headers and the sequences.
    '''
    with open(file, 'r') as handle:
        for line in handle:
            if line.startswith('>'):
                header = line[1:].strip()
                sequence = handle.readline().strip()
                yield header, sequence