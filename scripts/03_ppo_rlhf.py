from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch

POLICY_PATH = "checkpoints/sft-tinyllama"
RM_PATH     = "checkpoints/reward-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(POLICY_PATH, use_fast=True)
tok.pad_token = tok.eos_token

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_PATH, load_in_4bit=True, device_map="auto"
)
ref_model = AutoModelForCausalLM.from_pretrained(
    POLICY_PATH, load_in_4bit=True, device_map="auto"
)
rm = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH, load_in_4bit=True, device_map="auto"
)

cfg = PPOConfig(
    model_name=POLICY_PATH,
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    target_kl=0.1
)

ppo = PPOTrainer(cfg, policy, ref_model, tok)

prompts = load_dataset("stanfordnlp/SHP", split="test[:2%]")["instruction"]
prompts = [p if isinstance(p,str) else "" for p in prompts]
prompts = [p for p in prompts if len(p)>0][:cfg.batch_size]

def rm_score(texts):
    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(policy.device)
    with torch.no_grad():
        logits = rm(**inputs).logits.squeeze(-1)
    return logits.detach().float()

for step in range(50):  # 先跑 50 个 PPO updates 看曲线
    batch = prompts  # 简化：固定一批
    inputs = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(policy.device)
    gen = policy.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=1.0)
    responses = tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    rewards = rm_score([b + r for b, r in zip(batch, responses)])
    stats = ppo.step(batch, responses, rewards)
    ppo.log_stats(stats, batch, responses, rewards)

policy.save_pretrained("checkpoints/ppo-tinyllama")
