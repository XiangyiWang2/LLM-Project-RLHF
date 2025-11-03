from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k"
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

ds = load_dataset("stanfordnlp/SHP", split="train[:5%]")

def map_pair(ex):
    prompt = ex["history"][-1]["human"] if "history" in ex else ex.get("instruction","")
    a, b = ex["response_0"], ex["response_1"]
    # label: which is preferred (0 或 1)
    pref = ex["labels"]
    chosen = a if pref==0 else b
    rejected = b if pref==0 else a
    return {
        "pos": f"Human: {prompt}\nAssistant: {chosen}",
        "neg": f"Human: {prompt}\nAssistant: {rejected}"
    }

ds = ds.map(map_pair).remove_columns([c for c in ds.column_names if c not in ["pos","neg"]])
ds = ds.shuffle(seed=42).select(range(min(5000, len(ds))))

def explode(batch):
    pos = tok(batch["pos"], padding="max_length", truncation=True, max_length=512)
    neg = tok(batch["neg"], padding="max_length", truncation=True, max_length=512)
    return {
        "input_ids": pos["input_ids"] + neg["input_ids"],
        "attention_mask": pos["attention_mask"] + neg["attention_mask"],
        "labels": [1]*len(batch["pos"]) + [0]*len(batch["neg"])
    }

ds = ds.map(explode, batched=True, remove_columns=ds.column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, load_in_4bit=True, device_map="auto"
)

args = TrainingArguments(
    output_dir="runs/rm",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=20,
    fp16=True,
    evaluation_strategy="no",
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model("checkpoints/reward-model")
