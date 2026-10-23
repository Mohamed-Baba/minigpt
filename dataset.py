# dataset.py
import torch


class CharTokenizer:
    """Transforme du texte en numéros, et l'inverse. Un caractère = un token."""

    def __init__(self, text):
        chars = sorted(set(text))           # liste triée des caractères distincts
        # Remarque : Le tri est important pour conserver les codes des characters dans chaque lancement du code 
        self.vocab_size = len(chars)
        self.stoi = {ch : i for i,ch in enumerate(chars)}       # dict caractère -> numéro
        self.itos = {i : ch for i,ch in enumerate(chars)}       # dict numéro -> caractère

    def encode(self, s):
        # chaîne -> liste de numéros
        l = [self.stoi[c] for c in s]
        return l


    def decode(self, ids):
        # liste de numéros -> chaîne
        ch = ''.join([self.itos[i] for i in ids])
        return ch


class TextDataset(torch.utils.data.Dataset):
    """Comme SignalDataset : split='train' ou 'valid'.
    Un élément = une fenêtre x de block_size tokens, et y = x décalé d'un cran."""

    def __init__(self, text, tokenizer, split="train", block_size=8, train_frac=0.9):
        super().__init__()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(train_frac * len(data))
        if split == "train":
            self.data = data[:n]
        elif split == "valid":
            self.data = data[n:]
        else:
            raise ValueError(f"split doit être 'train' ou 'valid', reçu {split!r}")
        self.block_size = block_size

    def __len__(self):
        # nombre de positions de départ i valides
        return len(self.data) - self.block_size    
        # Remarque : normalement il manque + 1 mais on calcule avec (block_size+1) car y occupe un token de plus 

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i+1 : i+1 + self.block_size]
        return x, y