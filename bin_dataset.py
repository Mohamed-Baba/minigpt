# bin_dataset.py
# Lit un fichier .bin produit par prepare_data.py, SANS le charger en mémoire (np.memmap).
import numpy as np
import torch


class BinDataset(torch.utils.data.Dataset):
    """Comme TextDataset, mais les tokens restent sur le disque.
    Un élément = une fenêtre x de block_size tokens, et y = x décalé d'un cran."""

    def __init__(self, bin_path, block_size):
        super().__init__()
        # memmap : le fichier est « vu » comme un tableau, mais rien n'est lu tant qu'on ne demande rien
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        # x et y couvrent ensemble block_size + 1 tokens : N - (T+1) + 1 = N - T fenêtres
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        # on ne lit sur le disque que les block_size + 1 tokens nécessaires
        chunk = torch.from_numpy(self.data[i : i + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]          # x, y : [T] chacun, de type long