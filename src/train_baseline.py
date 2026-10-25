"""Tier 1: char n-gram TF-IDF + LogisticRegression."""
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from config import MODELS, PREDS, TABLES
from data import load

p = argparse.ArgumentParser()
p.add_argument("--view", default="translit", choices=["translit", "native"])
p.add_argument("--ngram_max", type=int, default=4)
args = p.parse_args()

df, col, labels, l2i = load(args.view)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))

vec = TfidfVectorizer(analyzer="char", ngram_range=(2, args.ngram_max),
                      min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(tr[col]); Xva = vec.transform(va[col]); Xte = vec.transform(te[col])

clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0, n_jobs=-1)
clf.fit(Xtr, tr.y)

for name, X, d in [("val", Xva, va), ("test", Xte, te)]:
    pred = clf.predict(X)
    print(f"[{name}] macro-F1 = {f1_score(d.y, pred, average='macro'):.4f}")

pred = clf.predict(Xte)
print(classification_report(te.y, pred, target_names=labels, digits=3, zero_division=0))

pd.DataFrame({"word": te[col].values, "y_true": te.y.values, "y_pred": pred}) \
  .to_csv(PREDS / f"baseline_{args.view}.csv", index=False)

feats = np.array(vec.get_feature_names_out())
rows = []
for i, lab in enumerate(labels):
    top = np.argsort(clf.coef_[i])[::-1][:25]
    rows.append({"label": lab, "top_ngrams": ", ".join(feats[top])})
pd.DataFrame(rows).to_csv(TABLES / f"baseline_top_ngrams_{args.view}.csv", index=False)

joblib.dump({"vec": vec, "clf": clf, "labels": labels}, MODELS / f"baseline_{args.view}.joblib")
json.dump({"macro_f1_test": float(f1_score(te.y, pred, average="macro"))},
          open(TABLES / f"baseline_{args.view}_metrics.json", "w"), indent=2)
