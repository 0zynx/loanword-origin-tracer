import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=64, filters=128, kernels=(2, 3, 4, 5), drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(filters * len(kernels), n_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, dim=1)))


class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=64, hidden=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden * 4, n_classes)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        pooled = torch.cat([h.max(1).values, h.mean(1)], dim=1)
        return self.fc(self.drop(pooled))


def build(arch, vocab_size, n_classes):
    return CharCNN(vocab_size, n_classes) if arch == "cnn" else CharBiLSTM(vocab_size, n_classes)
