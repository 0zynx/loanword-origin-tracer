"""Download Kaikki per-language dumps."""
import gzip, shutil, sys
import requests
from tqdm import tqdm
from config import RAW, LANGS

PATTERNS = [
    ("https://kaikki.org/dictionary/{Name}/kaikki.org-dictionary-{Name}.jsonl", False),
    ("https://kaikki.org/dictionary/{name}/kaikki.org-dictionary-{name}.jsonl", False),
    ("https://kaikki.org/dictionary/downloads/{code}/{code}.jsonl.gz",          True),
]

def fetch(name, code, out):
    for url_tpl, is_gz in PATTERNS:
        url = url_tpl.format(Name=name, name=name.lower(), code=code)
        try:
            r = requests.get(url, stream=True, timeout=90)
            if r.status_code != 200:
                print(f"  try {url} -> {r.status_code}")
                continue
            total = int(r.headers.get("content-length", 0))
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                            desc=name) as bar:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); bar.update(len(chunk))
            if is_gz:
                tmp.replace(out)
            else:
                with open(tmp, "rb") as fi, gzip.open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                tmp.unlink()
            print(f"[ok  ] {name} <- {url}")
            return True
        except Exception as e:
            print(f"  fail {url}: {e}")
    print(f"[MISS] {name}: check https://kaikki.org/dictionary/rawdata.html")
    return False

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(LANGS.values())
    inv = {v: k for k, v in LANGS.items()}
    for n in wanted:
        out = RAW / f"{n}.jsonl.gz"
        if out.exists():
            print(f"[skip] {out.name}"); continue
        fetch(n, inv[n], out)
