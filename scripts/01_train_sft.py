from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k"
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

# 小样本起步，足够当 PPO 的初始化
ds = load_dataset("OpenAssistant/oasst1", split="train[:2%]")
# oasst1 字段很多，用 text 字段或简单拼接；这里直接用现成 text（如无则跳过）
if "text" not in ds.column_names:
    def to_text(ex):
        msg = ex.get("text") or ex.get("message") or ""
        return {"text": msg}
    ds = ds.map(to_text)
    ds = ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"])>0)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(BASE, load_in_4bit=True, device_map="auto")
args = TrainingArguments(
    output_dir="runs/sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=20,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True
)

trainer = SFTTrainer(
    model=model, tokenizer=tok, peft_config=peft_cfg,
    train_dataset=ds, dataset_text_field="text", args=args,
    packing=True, max_seq_length=1024
)
trainer.train()
trainer.save_model("checkpoints/sft-tinyllama")
