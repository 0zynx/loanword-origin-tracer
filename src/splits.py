"""Group-aware 70/15/15 split so cognates don't leak across splits."""
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import PROCESSED, SEED

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED / "labeled.csv")

    # pandas turns empty strings into NaN on read — fill them back
    df["word_translit"] = df["word_translit"].fillna("")
    df["word_native"] = df["word_native"].fillna("")

    # drop rows where BOTH views are empty (unusable)
    df = df[(df["word_translit"] != "") | (df["word_native"] != "")].copy()
    print(f"kept {len(df):,} usable rows")

    labels = sorted(df["label"].unique())
    json.dump(labels, open(PROCESSED / "labels.json", "w"), indent=2)

    df["group"] = df["word_translit"].str[:3] + "_" + df["label"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, rest_idx = next(gss.split(df, groups=df["group"]))
    tr, rest = df.iloc[tr_idx], df.iloc[rest_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_idx, te_idx = next(gss2.split(rest, groups=rest["group"]))
    va, te = rest.iloc[va_idx], rest.iloc[te_idx]

    tr["split"], va["split"], te["split"] = "train", "val", "test"
    out = pd.concat([tr, va, te]).sort_index()
    out.to_csv(PROCESSED / "splits.csv", index=False)
    print(f"labels: {labels}")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))