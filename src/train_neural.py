"""Tier 2: char CNN / BiLSTM."""
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from config import CFG, MODELS, PREDS, TABLES, SEED
from data import load, CharVocab, CharDataset, class_weights
import models

p = argparse.ArgumentParser()
p.add_argument("--view", default="translit", choices=["translit", "native"])
p.add_argument("--arch", default="cnn", choices=["cnn", "lstm"])
args = p.parse_args()

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df, col, labels, l2i = load(args.view)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
vocab = CharVocab(tr[col])

dl = lambda d, sh: DataLoader(CharDataset(d[col], d.y, vocab), batch_size=CFG["neural"]["batch_size"], shuffle=sh)
tr_dl, va_dl, te_dl = dl(tr, True), dl(va, False), dl(te, False)

model = models.build(args.arch, len(vocab), len(labels)).to(dev)
w = class_weights(tr.y.values, len(labels)).to(dev)
lossf = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), lr=CFG["neural"]["lr"])

def run(loader):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P += model(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
    return np.array(Y), np.array(P)

best, patience = -1, 0
for ep in range(CFG["neural"]["epochs"]):
    model.train()
    for x, y in tr_dl:
        opt.zero_grad(); loss = lossf(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step()
    yv, pv = run(va_dl); f1 = f1_score(yv, pv, average="macro")
    print(f"epoch {ep+1:02d}  val macro-F1 = {f1:.4f}")
    if f1 > best:
        best, patience = f1, 0
        torch.save(model.state_dict(), MODELS / f"{args.arch}_{args.view}.pt")
    else:
        patience += 1
        if patience >= CFG["neural"]["patience"]:
            break

model.load_state_dict(torch.load(MODELS / f"{args.arch}_{args.view}.pt"))
yt, pt = run(te_dl)
print(classification_report(yt, pt, target_names=labels, digits=3, zero_division=0))
pd.DataFrame({"word": te[col].values, "y_true": yt, "y_pred": pt}) \
  .to_csv(PREDS / f"{args.arch}_{args.view}.csv", index=False)
json.dump({"macro_f1_test": float(f1_score(yt, pt, average="macro"))},
          open(TABLES / f"{args.arch}_{args.view}_metrics.json", "w"), indent=2)
