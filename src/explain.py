"""Occlusion saliency for any tier: drop each char, measure confidence drop."""
import argparse, json, joblib
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from config import MODELS, TABLES
from data import load, CharVocab
import models

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True,
               choices=["baseline", "cnn", "lstm", "muril", "xlmr"])
p.add_argument("--view", default="translit", choices=["translit", "native"])
p.add_argument("--n", type=int, default=300)
a = p.parse_args()

df, col, labels, _ = load(a.view)
te = df[df.split == "test"].head(a.n)

def baseline():
    b = joblib.load(MODELS / f"baseline_{a.view}.joblib")
    return lambda ws: b["clf"].predict_proba(b["vec"].transform(ws)), b["labels"]

def neural(kind):
    vocab = CharVocab(df[df.split == "train"][col])
    m = models.build(kind, len(vocab), len(labels))
    m.load_state_dict(torch.load(MODELS / f"{kind}_{a.view}.pt", map_location="cpu"))
    m.eval()
    def f(ws):
        X = torch.tensor([vocab.encode(w) for w in ws])
        with torch.no_grad():
            return torch.softmax(m(X), 1).numpy()
    return f, labels

def transformer():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = MODELS / f"{a.kind}_{a.view}"
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    labs = [m.config.id2label[i] for i in range(len(labels))]
    def f(ws):
        with torch.no_grad():
            enc = tok(ws, return_tensors="pt", padding=True,
                      truncation=True, max_length=32)
            return torch.softmax(m(**enc).logits, 1).numpy()
    return f, labs

if a.kind == "baseline":
    fn, labs = baseline()
elif a.kind in ("cnn", "lstm"):
    fn, labs = neural(a.kind)
else:
    fn, labs = transformer()

rows = []
for w in tqdm(te[col]):
    base = fn([w]).max()
    drops = [base - fn([w[:i] + w[i+1:]]).max() for i in range(len(w))]
    k = int(np.argmax(drops)) if drops else 0
    rows.append({"word": w, "pred": labs[fn([w]).argmax()],
                 "top_char": w[k] if w else "",
                 "context": w[max(0, k-2):k+3],
                 "drop": float(drops[k]) if drops else 0.0})

out = pd.DataFrame(rows)
out.to_csv(TABLES / f"saliency_{a.kind}_{a.view}.csv", index=False)
print(out.sort_values("drop", ascending=False).head(20).to_string(index=False))
