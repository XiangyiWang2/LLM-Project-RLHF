import os, json, torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SFT_PATH = "checkpoints/sft-tinyllama"
PPO_PATH = "checkpoints/ppo-tinyllama"
RM_PATH  = "checkpoints/reward-model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

PROMPTS = [
    "Explain why the sky appears blue in a clear day.",
    "Give me a short study plan for a data structures exam.",
    "你是如何高效整理课堂笔记的？给 3 条建议。",
    "Summarize pros and cons of running 20 minutes daily.",
    "写一段 Python 代码，读取 CSV 并打印均值。",
    "What’s the intuition behind gradient descent?",
    "如何解释数据库中的外键及其 ON DELETE CASCADE？",
    "Give two practical tips to improve sleep quality."
]

os.makedirs("figures", exist_ok=True)

def load_tok():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE)

def load_causal_with_adapter(adapter_dir, tok):
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE)
    base.config.use_cache = False
    m = PeftModel.from_pretrained(base, adapter_dir).eval()
    return m

def load_rm():
    tok_rm = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    added = False
    if tok_rm.pad_token is None:
        tok_rm.add_special_tokens({"pad_token": "[PAD]"}); added = True
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=1, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE
    )
    if added: rm_base.resize_token_embeddings(len(tok_rm))
    rm_base.config.pad_token_id = tok_rm.pad_token_id
    rm = PeftModel.from_pretrained(rm_base, RM_PATH).eval()
    for p in rm.parameters(): p.requires_grad_(False)
    return tok_rm, rm

@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=128, top_p=0.9, temperature=1.0):
    inputs = tok([f"Human: {p}\nAssistant:" for p in prompts], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p, temperature=temperature,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
    )
    return tok.batch_decode(out, skip_special_tokens=True)

@torch.no_grad()
def score_rm(texts, tok_rm, rm):
    enc = tok_rm(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(rm.device) for k, v in enc.items()}
    s = rm(**enc).logits.view(-1).float().detach().cpu().tolist()
    return s

def main():
    tok = load_tok()
    models = {}
    if os.path.isfile(os.path.join(SFT_PATH, "adapter_config.json")):
        models["SFT"] = load_causal_with_adapter(SFT_PATH, tok)
    if os.path.isfile(os.path.join(PPO_PATH, "adapter_config.json")):
        models["PPO"] = load_causal_with_adapter(PPO_PATH, tok)
    else:
        print("⚠️ 未找到 PPO 适配器（checkpoints/ppo-tinyllama/adapter_config.json），将只评测 SFT。")

    tok_rm, rm = load_rm()

    results = {}
    for name, m in models.items():
        texts = generate(m, tok, PROMPTS, max_new_tokens=128, top_p=0.9, temperature=1.0)
        scores = score_rm(texts, tok_rm, rm)
        results[name] = {"texts": texts, "scores": scores}

    # 画图：RM 平均分对比
    labels = list(results.keys())
    means  = [sum(v["scores"])/max(1,len(v["scores"])) for v in results.values()]
    plt.figure(figsize=(5,4))
    plt.bar(labels, means)
    plt.ylabel("Reward Model score (higher is better)")
    plt.title("SFT vs PPO — RM scoring")
    plt.tight_layout()
    fig_path = "figures/ppo_vs_sft_rm.png"
    plt.savefig(fig_path, dpi=200)
    print(f"[OK] 图已保存：{fig_path}")

    # 保存详细样本
    dump_path = "figures/ppo_samples.jsonl"
    with open(dump_path, "w", encoding="utf-8") as f:
        for name, v in results.items():
            for p, t, s in zip(PROMPTS, v["texts"], v["scores"]):
                f.write(json.dumps({"model": name, "prompt": p, "text": t, "rm": s}, ensure_ascii=False)+"\n")
    print(f"[OK] 详细样本已保存：{dump_path}")

if __name__ == "__main__":
    main()