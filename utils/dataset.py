import torch
from tqdm.notebook import trange, tqdm

__version__ = "1.0.0"

PAD = "<pad>" # Padding
SOS = "<SOS>" # Start Of Sequence
EOS = "<EOS>" # End Of Sequence

class CityNameDataset():
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