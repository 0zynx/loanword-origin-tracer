"""Comparison table, per-class F1, named confusion matrices."""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from config import PROCESSED, PREDS, TABLES, FIGS

labels = json.load(open(PROCESSED / "labels.json"))
rows, perclass = [], []

for f in sorted(glob.glob(str(PREDS / "*.csv"))):
    d = pd.read_csv(f); name = Path(f).stem
    rows.append({"model": name,
                 "accuracy": accuracy_score(d.y_true, d.y_pred),
                 "macro_f1": f1_score(d.y_true, d.y_pred, average="macro"),
                 "weighted_f1": f1_score(d.y_true, d.y_pred, average="weighted")})
    p, r, f1, s = precision_recall_fscore_support(
        d.y_true, d.y_pred, labels=range(len(labels)), zero_division=0)
    for i, lab in enumerate(labels):
        perclass.append({"model": name, "label": lab, "precision": p[i],
                         "recall": r[i], "f1": f1[i], "support": int(s[i])})
    cm = confusion_matrix(d.y_true, d.y_pred, labels=range(len(labels)))
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(name); plt.xlabel("predicted"); plt.ylabel("true")
    plt.tight_layout(); plt.savefig(FIGS / f"cm_{name}.png", dpi=200); plt.close()

pd.DataFrame(rows).sort_values("macro_f1", ascending=False) \
  .to_csv(TABLES / "model_comparison.csv", index=False)
pd.DataFrame(perclass).to_csv(TABLES / "per_class_f1.csv", index=False)
print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_string(index=False))
