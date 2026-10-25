"""Tamil case study: borrows, lends, Tamil-row accuracy."""
import glob, json
from pathlib import Path
import pandas as pd
from config import PROCESSED, PREDS, TABLES

df = pd.read_csv(PROCESSED / "splits.csv")
labels = json.load(open(PROCESSED / "labels.json"))

tam = df[df.lang_name == "Tamil"]
inbound = tam[tam.label != "native"]["label"].value_counts().rename("count")
outbound = df[(df.label == "Tamil") & (df.lang_name != "Tamil")]["lang_name"].value_counts().rename("count")

inbound.to_csv(TABLES / "tamil_inbound.csv")
outbound.to_csv(TABLES / "tamil_outbound.csv")
print("Tamil borrows FROM:\n", inbound.to_string(), "\n")
print("Others borrow FROM Tamil:\n", outbound.to_string(), "\n")

rows = []
for f in sorted(glob.glob(str(PREDS / "*.csv"))):
    d = pd.read_csv(f)
    y = df.loc[d.index]
    m = (y.lang_name == "Tamil") | (y.label == "Tamil")
    if m.sum() == 0:
        continue
    sub = d[m.values]
    rows.append({"model": Path(f).stem,
                 "tamil_rows": int(len(sub)),
                 "tamil_accuracy": float((sub.y_true == sub.y_pred).mean())})
pd.DataFrame(rows).to_csv(TABLES / "tamil_accuracy.csv", index=False)
print("\nTamil-row accuracy:\n", pd.DataFrame(rows).to_string(index=False))
