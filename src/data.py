import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import PROCESSED

PAD, UNK = 0, 1


def load(view="translit"):
    col = "word_translit" if view == "translit" else "word_native"
    df = pd.read_csv(PROCESSED / "splits.csv")
    df[col] = df[col].fillna("")          # <-- the fix
    labels = sorted(df["label"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    df["y"] = df["label"].map(l2i)
    return df, col, labels, l2i


class CharVocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self): return len(self.itos)
    def encode(self, s, maxlen=40):
        ids = [self.stoi.get(c, UNK) for c in s][:maxlen]
        return ids + [PAD] * (maxlen - len(ids))


class CharDataset(Dataset):
    def __init__(self, texts, ys, vocab, maxlen=40):
        self.X = torch.tensor([vocab.encode(t, maxlen) for t in texts], dtype=torch.long)
        self.y = torch.tensor(list(ys), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float)