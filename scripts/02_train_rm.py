import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, Value
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
added_pad = False
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
    added_pad = True
tok.padding_side = "right"
pad_id = tok.pad_token_id


ds = load_dataset("stanfordnlp/SHP", split="train").shuffle(seed=42).select(range(3000))

def map_pair(ex):
    prompt = ex.get("history", "")
    a = ex.get("human_ref_A", "")
    b = ex.get("human_ref_B", "")
    chosen, rejected = (a, b) if ex.get("labels", 1) == 1 else (b, a)
    return {
        "pos": f"Human: {prompt}\nAssistant: {chosen}",
        "neg": f"Human: {prompt}\nAssistant: {rejected}",
    }
ds = ds.map(map_pair)

def tokenize_batch(batch):
    pos = tok(batch["pos"], padding="max_length", truncation=True, max_length=512)
    neg = tok(batch["neg"], padding="max_length", truncation=True, max_length=512)

    return {
        "input_ids": pos["input_ids"] + neg["input_ids"],
        "attention_mask": pos["attention_mask"] + neg["attention_mask"],
        "labels": [1.0] * len(batch["pos"]) + [0.0] * len(batch["neg"]),
    }
ds = ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)

ds = ds.cast_column("labels", Value("float32"))


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, quantization_config=bnb, device_map="auto"
)
if added_pad:
    model.resize_token_embeddings(len(tok))

model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(model, peft_cfg)


model.config.pad_token_id = pad_id
model.config.problem_type = "regression"
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = pad_id
emb = model.get_input_embeddings()
if hasattr(emb, "padding_idx") and (emb.padding_idx is None or emb.padding_idx < 0):
    emb.padding_idx = pad_id

args = TrainingArguments(
    output_dir="runs/rm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=20,
    fp16=True,
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

trainer.save_model("checkpoints/reward-model")
tok.save_pretrained("checkpoints/reward-model")
print("[RM] saved to checkpoints/reward-model")
