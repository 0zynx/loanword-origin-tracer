import argparse, joblib
from config import MODELS
from preprocess import script_of, translit, clean

p = argparse.ArgumentParser()
p.add_argument("word")
p.add_argument("--view", default="translit")
a = p.parse_args()

b = joblib.load(MODELS / f"baseline_{a.view}.joblib")
vec, clf, labels = b["vec"], b["clf"], b["labels"]

w = clean(a.word) if a.view == "native" else clean(translit(a.word, script_of(a.word)))
proba = clf.predict_proba(vec.transform([w]))[0]
order = proba.argsort()[::-1]
print(f"input        : {w}")
for i in order[:5]:
    print(f"  {labels[i]:<12} {proba[i]:.3f}")

base = proba.max(); imp = []
for i in range(len(w)):
    p2 = clf.predict_proba(vec.transform([w[:i] + w[i + 1:]]))[0].max()
    imp.append(base - p2)
print("\nchar importance:")
for ch, s in zip(w, imp):
    print(f"  {ch}  {'#' * int(max(s, 0) * 200)}")
