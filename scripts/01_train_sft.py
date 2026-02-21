from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, peft_cfg)


ds = load_dataset("OpenAssistant/oasst1", split="train[:2%]")

def to_text(ex):
    t = ex.get("text")
    if isinstance(t, str) and len(t) > 0: return {"text": t}
    msg = ex.get("message") or ex.get("messages")
    if isinstance(msg, list):
        parts=[]
        for m in msg:
            role=m.get("role","")
            c=m.get("content","")
            if isinstance(c, dict): c=c.get("text","")
            if isinstance(c, list): c=" ".join([seg.get("text","") for seg in c if isinstance(seg, dict)])
            if c: parts.append(f"{role}: {c}")
        return {"text":"\n".join(parts)}
    return {"text": ""}

ds = ds.map(to_text)
ds = ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"])>0)

def tokenize(batch): return tok(batch["text"], truncation=True, max_length=1024)
tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

collator = DataCollatorForLanguageModeling(tok, mlm=False)
args = TrainingArguments(
    output_dir="runs/sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=20,
    fp16=True,
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=collator)
trainer.train()

os.makedirs("checkpoints/sft-tinyllama", exist_ok=True)
trainer.save_model("checkpoints/sft-tinyllama")
tok.save_pretrained("checkpoints/sft-tinyllama")
print("[SFT] saved to checkpoints/sft-tinyllama")
