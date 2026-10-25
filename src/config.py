from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUT = ROOT / "outputs"
MODELS = OUT / "models"
PREDS = OUT / "preds"
TABLES = OUT / "tables"
FIGS = OUT / "figures"

for p in [RAW, INTERIM, PROCESSED, MODELS, PREDS, TABLES, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

LANGS = CFG["languages"]
TARGETS = CFG["target_labels"]
SEED = CFG["seed"]
