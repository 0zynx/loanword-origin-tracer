"""Turn Kaikki JSONL into a flat table with source-language labels."""
import gzip, json, re, sys
import pandas as pd
from tqdm import tqdm
from config import CFG, RAW, INTERIM, LANGS, TARGETS

BORROW = {"bor", "borrowed", "lbor", "learned borrowing", "slbor",
          "semantic loan", "ubor", "obor"}
DERIVE = {"der", "derived", "root"}
INHERIT = {"inh", "inherited"}
CALQUE = {"cal", "calque", "clq"}

REL_MAP = {}
for s in BORROW:  REL_MAP[s] = "borrowed"
for s in DERIVE:  REL_MAP[s] = "derived"
for s in INHERIT: REL_MAP[s] = "inherited"
for s in CALQUE:  REL_MAP[s] = "calque"

NAME2CODE = {v.lower(): k for k, v in LANGS.items()}
NAME2CODE.update({"old tamil": "ta", "middle tamil": "ta",
                  "hindi": "hi", "urdu": "hi",
                  "old persian": "fa", "middle persian": "fa",
                  "classical persian": "fa", "arabic": "ar",
                  "classical arabic": "ar", "portuguese": "pt",
                  "old portuguese": "pt", "english": "en",
                  "old english": "en", "middle english": "en",
                  "sanskrit": "sa", "vedic sanskrit": "sa"})

FALLBACK = re.compile(r"(?:Borrowed|Derived|Inherited) from ([A-Z][A-Za-z\- ]{2,30})")


def from_templates(obj):
    lang = obj.get("lang_code")
    for t in obj.get("etymology_templates") or []:
        name = (t.get("name") or "").strip().lower()
        if name not in REL_MAP:
            continue
        args = t.get("args") or {}
        if str(args.get("1", "")).strip() != lang:
            continue
        src = str(args.get("2", "")).strip()
        if src and src != lang:
            return src, REL_MAP[name]
    return None, None


def from_text(obj):
    text = obj.get("etymology_text") or ""
    m = FALLBACK.search(text)
    if not m:
        return None, None
    name = m.group(1).strip().lower()
    code = NAME2CODE.get(name)
    if not code:
        return None, None
    rel = "inherited" if text.lower().startswith("inherited") else "borrowed"
    return code, rel


def to_label(src_code, rel):
    if rel == "inherited":
        return "native"
    if rel in ("borrowed", "derived") and src_code in LANGS:
        name = LANGS[src_code]
        return name if name in TARGETS else CFG["other_label"]
    return None


def extract_one(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=path.name.split(".")[0]):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = (obj.get("word") or "").strip()
            lang = obj.get("lang_code")
            if not word or lang not in LANGS:
                continue
            src, rel = from_templates(obj)
            if not src:
                src, rel = from_text(obj)
            if not src:
                continue
            label = to_label(src, rel)
            if not label:
                continue
            rows.append({
                "word": word,
                "lang_code": lang,
                "lang_name": LANGS[lang],
                "pos": obj.get("pos", ""),
                "source_code": src,
                "source_name": LANGS.get(src, src),
                "relation": rel,
                "label": label,
                "etymology_text": (obj.get("etymology_text") or "")[:400],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    names = sys.argv[1:] or list(LANGS.values())
    frames = []
    for n in names:
        p = RAW / f"{n}.jsonl.gz"
        if not p.exists():
            print(f"[skip] {p.name} not downloaded")
            continue
        frames.append(extract_one(p))
    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(subset=["word", "lang_code", "source_code", "relation"], inplace=True)
    out = INTERIM / "raw_edges.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df):,} rows -> {out}")
    print(df["label"].value_counts())
