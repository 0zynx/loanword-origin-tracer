"""Tier 3: fine-tune MuRIL / XLM-R with class-weighted loss."""
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from config import CFG, MODELS, PREDS, TABLES, SEED
from data import load, class_weights

p = argparse.ArgumentParser()
p.add_argument("--view", default="translit", choices=["translit", "native"])
p.add_argument("--model", default=CFG["transformer"]["primary"])
p.add_argument("--tag", default="muril")
args = p.parse_args()

df, col, labels, l2i = load(args.view)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
tok = AutoTokenizer.from_pretrained(args.model)

class DS(torch.utils.data.Dataset):
    def __init__(self, d):
        self.enc = tok(list(d[col]), truncation=True, padding="max_length",
                       max_length=CFG["transformer"]["max_len"])
        self.y = list(d.y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i]); return item

tr_ds, va_ds, te_ds = DS(tr), DS(va), DS(te)
cw = class_weights(tr.y.values, len(labels))

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels_t = inputs.pop("labels")
        out = model(**inputs)
        loss = F.cross_entropy(out.logits, labels_t, weight=cw.to(out.logits.device))
        return (loss, out) if return_outputs else loss

def metrics(ep):
    pred = np.argmax(ep.predictions, axis=1)
    return {"macro_f1": f1_score(ep.label_ids, pred, average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels),
    id2label={i: l for l, i in l2i.items()}, label2id=l2i)

targs = TrainingArguments(
    output_dir=str(MODELS / f"{args.tag}_{args.view}"),
    learning_rate=CFG["transformer"]["lr"],
    per_device_train_batch_size=CFG["transformer"]["batch_size"],
    per_device_eval_batch_size=64,
    num_train_epochs=CFG["transformer"]["epochs"],
    weight_decay=0.01, seed=SEED, logging_steps=100,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="macro_f1",
    greater_is_better=True, report_to="none", save_total_limit=1)

trainer = WTrainer(model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
                   compute_metrics=metrics)
trainer.train()

out = trainer.predict(te_ds)
pred = np.argmax(out.predictions, axis=1)
print(classification_report(out.label_ids, pred, target_names=labels, digits=3, zero_division=0))
pd.DataFrame({"word": te[col].values, "y_true": out.label_ids, "y_pred": pred}) \
  .to_csv(PREDS / f"{args.tag}_{args.view}.csv", index=False)
json.dump({"macro_f1_test": float(f1_score(out.label_ids, pred, average="macro"))},
          open(TABLES / f"{args.tag}_{args.view}_metrics.json", "w"), indent=2)
trainer.save_model(MODELS / f"{args.tag}_{args.view}")
tok.save_pretrained(MODELS / f"{args.tag}_{args.view}")
