"""NFC normalise, build Latin transliteration view, filter rare classes."""
import re
import pandas as pd
import unicodedata
from config import CFG, INTERIM, PROCESSED

TAMIL = range(0x0B80, 0x0C00)
DEVAN = range(0x0900, 0x0980)
ARABIC = list(range(0x0600, 0x0700)) + list(range(0x0750, 0x0780)) + \
         list(range(0xFB50, 0xFE00)) + list(range(0xFE70, 0xFF00))


def script_of(word):
    for ch in word:
        cp = ord(ch)
        if cp in TAMIL:   return "tamil"
        if cp in DEVAN:   return "devanagari"
        if cp in ARABIC:  return "arabic"
        if ch.isalpha():  return "latin"
    return "other"


def translit(word, script):
    try:
        if script == "tamil":
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            return transliterate(word, sanscript.TAMIL, sanscript.IAST)
        if script == "devanagari":
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            return transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)
        if script == "arabic":
            from unidecode import unidecode
            return unidecode(word)
    except Exception:
        pass
    return word


def clean(s):
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = re.sub(r"[^\w\u0B80-\u0BFF\u0900-\u097F\u0600-\u06FF]+", "", s)
    return s


if __name__ == "__main__":
    df = pd.read_csv(INTERIM / "raw_edges.csv")
    df["script"] = df["word"].map(script_of)
    df["word_native"] = df["word"].map(clean)
    df["word_translit"] = [clean(translit(w, s)) for w, s in zip(df["word"], df["script"])]
    df = df[(df.word_native.str.len() >= CFG["min_word_len"]) &
            (df.word_native.str.len() <= CFG["max_word_len"])]
    counts = df["label"].value_counts()
    keep = counts[counts >= CFG["min_class_count"]].index
    df = df[df["label"].isin(keep)].copy()
    df.to_csv(PROCESSED / "labeled.csv", index=False)
    print(df["label"].value_counts())
    print(df.groupby(["lang_name", "label"]).size().unstack(fill_value=0))
