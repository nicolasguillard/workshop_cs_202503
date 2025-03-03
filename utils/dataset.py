import random
from typing import Tuple
import torch
from torch.utils.data import Dataset
from tqdm.notebook import trange, tqdm

__version__ = "1.1.1"

SOS = "<SOS>" # Start Of Sequence
EOS = "<EOS>" # End Of Sequence
PAD = "<PAD>" # Padding

class CityNameDataset_V1():
    def __init__(self, file_name: str = "villes.txt", split_rate: float =0.9, device="cpu"):

        # chargement des données
        fichier = open(file_name)
        donnees = fichier.read()
        villes = donnees.replace('\n', ',').split(',')
        # mise à l'écart des villes avec un nom de longueur inférieure à 3
        self.villes = [ville for ville in villes if len(ville) > 2]

        # création du vocabulaire
        self.vocabulaire = sorted(list(set(''.join(villes))))
        self.vocabulaire = [PAD, SOS, EOS] + self.vocabulaire
        # <SOS> et <EOS> sont ajoutés respectivement au début et à la fin de chaque séquence
        # <pad> est utilisé pour faire en sorte que toutes les séquences aient la même longueur

        # pour convertir char <-> int
        self.char_to_int = {}
        self.int_to_char = {}

        for (c, i) in tqdm(zip(self.vocabulaire, range(len(self.vocabulaire))), desc="creating vocabulary", total=len(self.vocabulaire)):
            self.char_to_int[c] = i
            self.int_to_char[i] = c

        num_sequences = len(villes)
        self.max_len = max([len(ville) for ville in villes]) + 2 # <SOS> et <EOS>

        X = torch.zeros((num_sequences, self.max_len), dtype=torch.int32, device=device)

        # création des séquences
        for i in trange(num_sequences, desc="creatind dataset"):
            X[i] = torch.tensor([self.char_to_int[SOS]] +
                                [self.char_to_int[c] for c in villes[i]] +
                                [self.char_to_int[EOS]] +
                                [self.char_to_int[PAD]] * (self.max_len - len(villes[i]) - 2))

        # jeu de données d'entrainement et de validation
        n_split = int(split_rate * X.shape[0])

        idx_permut = torch.randperm(X.shape[0])
        idx_train, _ = torch.sort(idx_permut[:n_split])
        idx_val, _ = torch.sort(idx_permut[n_split:])

        self.X_train = X[idx_train]
        self.X_val = X[idx_val]

    def get_batch(self, batch_size: int, split : str="val", device=None) -> torch.Tensor:
        assert split in ["train", "val"], f"split ({split}) should be 'train' or 'val'." 

        data = self.X_train if split == 'train' else self.X_val

        idx = torch.randint(low=int(batch_size/2), high=int(data.shape[0]-batch_size/2), size=(1,), dtype=torch.int32).item()

        batch = data[int(idx-batch_size/2):int(idx+batch_size/2)]
        X = batch[:, :-1] # (B, S=max_len-1) : (max(len("nom")) + 2) - 1
        Y = batch[:, 1:] # (B, S=max_len-1)
        if device:
            X = X.to(device)
            Y = Y.to(device)
        return X, Y.long()
    
    def cast_char_to_int(self, sequence: list[str]) -> list[int]:
        return [self.char_to_int[c] for c in sequence]
    
    def cast_int_to_char(self, sequence: list[int]) -> list[str]:
        return [self.int_to_char[i] for i in sequence]
    
    def to_string(self, sequence: list[int]) -> str:
        return "".join([self.int_to_char[i] for i in sequence if i > 2])


class CharTokenizer():
    def __init__(self, corpus: list[str]) -> None:
        self.special_tokens =  [PAD, SOS, EOS]

         # liste des différents caractères distincts
        chars = sorted(list(set(''.join(corpus))))
        # ajout des trois jetons de contrôle
        chars = self.special_tokens + chars
        # tables de correspondances
        self.char_to_int = {}
        self.int_to_char = {}

        # indexation des tables de correspondances
        for (c, i) in tqdm(
            zip(chars, range(len(chars))),
            desc="creating vocabulary",
            total=len(chars)
            ):
            self.char_to_int[c] = i
            self.int_to_char[i] = c

    def __call__(self, string: str) -> list[int]:
        return self.to_idx(string)
            
    def vocabulary_size(self) -> int:
        return len(self.char_to_int)
    
    def to_idx(self, sequence: str) -> list[int]:
        """
        Translate a sequence of chars to its conterparts of indexes in the vocabulary
        """
        return [self.char_to_int[c] for c in sequence]
    
    def to_tokens(self, sequence: list[int]) -> list[str]:
        """
        Translate a sequence of indexes to its conterparts of chars in the vocabulary
        """
        return [self.int_to_char[i] for i in sequence]
    
    def to_string(self, sequence: list[int]) -> str:
        """
        Return the string corresponding to the sequence of indexes in the vocabulary
        """
        return "".join([self.int_to_char[i] for i in sequence if i > 2])


class CityNameDataset(Dataset):
    def __init__(self, names: list[str], tokenizer: CharTokenizer) -> None:
        """
        Args:
            - names : collection of string
            - vocabulary : maps of "char to index" and "index to char" based on names
        """
        super().__init__()

        self.tokenizer = tokenizer
        
        # création des séquences encodées
        num_sequences = len(names)
        self.max_len = max([len(name) for name in names]) + 2 # <SOS> et <EOS>
        self.X = torch.zeros((num_sequences, self.max_len), dtype=torch.int32)
        for i, name in tqdm(enumerate(names), total=num_sequences, desc="creatind dataset"):
            # encodage de la séquence : "SOS s e q u e n c e EPS PAD PAD ... PAD"
            self.X[i] = torch.tensor(
                self.tokenizer([SOS]) +
                self.tokenizer(name) +
                self.tokenizer([EOS]) +
                self.tokenizer([PAD] * (self.max_len - len(name) - 2))
            )
            
    def __len__(self):
        """
        """
        return self.X.size(0)

    def __getitem__(self, idx: int):
        """
        """
        return self.X[idx, :-1], self.X[idx, 1:]


def get_vocabulary(corpus: list[str]) -> Tuple[dict, dict]:
    """
    Process the maps of "char to index" and "index to char" based on corpus

    Args:
        - corpus (list[str]) : collection of strings to map on.

    Returns:
        - char_to_int, int_to_char (Tuple[dict, dict]) : the mappings
    """
    
    # liste des différents caractères distincts
    chars = sorted(list(set(''.join(corpus))))
    # ajout des trois jetons de contrôle
    chars = [PAD, SOS, EOS] + chars
    # tables de correspondances
    char_to_int = {}
    int_to_char = {}

    # indexation des tables de correspondances
    for (c, i) in tqdm(
        zip(chars, range(len(chars))),
        desc="creating vocabulary",
        total=len(chars)
        ):
        char_to_int[c] = i
        int_to_char[i] = c
    return char_to_int, int_to_char


def get_datasets(
        filename: str,
        split_rate: float = 0.9
        ) -> Tuple[CityNameDataset, CityNameDataset, CharTokenizer, int]:
    """
    Return train and test datasets, and the max length in the processed string collection

    Args:
        - filename (str) : path and file name of string data
        - split_rate (float) : rate of the split of the train data, in [0.; 1.]
    
    Returns:
        - train_dataset, test_dataset, max_len (Tuple[CityNameDataset, CityNameDataset, int]) : 
            dataset for train, dataset for test, max length in the processed string collection
    """
    # chargement des données
    file = open(filename)
    raws = file.read()
    names = raws.replace('\n', ',').split(',')
    # mise à l'écart des villes avec un nom de longueur inférieure à 3
    # et détermination de la plus grande longueur
    names_ = []
    max_len = 0
    for n in names:
        if len(names) > 2:
            names_.append(n)
            max_len = max(max_len, len(n))

    # création du tokenizer
    tokenizer = CharTokenizer(names_)

    # mélange de l'ensemble des noms
    names_ = random.sample(names_, len(names_))
    # index de séparation train/val
    n_split = int(split_rate * len(names_))
    
    train_dataset = CityNameDataset(names_[:n_split], tokenizer)
    test_dataset = CityNameDataset(names_[n_split:], tokenizer)

    return train_dataset, test_dataset, tokenizer, max_len