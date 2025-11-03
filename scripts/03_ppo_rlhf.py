import os, random, math, json, time
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# ----------------------------
# 基本配置
# ----------------------------
BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SFT_PATH = "checkpoints/sft-tinyllama"
RM_PATH  = "checkpoints/reward-model"
OUT_DIR  = "checkpoints/ppo-tinyllama"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 64
TOP_P = 0.9
TEMPERATURE = 1.0

BATCH_SIZE = 1           # 小批次稳定点
UPDATES    = 500           # 训练步数（可加大，比如 200）
CLIP_EPS   = 0.02
LR         = 1e-6
VF_COEF    = 0.5
ENT_COEF   = 0.0
KL_COEF    = 0.1         # 轻微 KL 惩罚

random.seed(42)
torch.manual_seed(42)

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Tokenizer (Policy)
# ----------------------------
tok_pol = AutoTokenizer.from_pretrained(BASE, use_fast=False)
if tok_pol.pad_token is None:
    tok_pol.pad_token = tok_pol.eos_token
tok_pol.padding_side = "left"

# ----------------------------
# Tokenizer (Reward Model)
# ----------------------------
tok_rm = AutoTokenizer.from_pretrained(BASE, use_fast=False)
added_pad_to_rm = False
if tok_rm.pad_token is None:
    tok_rm.add_special_tokens({"pad_token": "[PAD]"})
    added_pad_to_rm = True
tok_rm.padding_side = "right"


# ----------------------------
# 加载 policy（SFT 作为初始化）与 reference（冻结，仅算 KL）
# ----------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE)

def load_policy_with_lora():
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE)
    base.config.use_cache = False
    pol = PeftModel.from_pretrained(base, SFT_PATH)
    return pol

policy = load_policy_with_lora()
ref_model = load_policy_with_lora().eval()
for p in ref_model.parameters(): p.requires_grad_(False)

# ----------------------------
# Value Head
# ----------------------------
HIDDEN = policy.base_model.model.model.embed_tokens.embedding_dim
# [FIX 5.1] 将 value_head 显式转换为 DTYPE (float16)
value_head = nn.Linear(HIDDEN, 1, bias=True).to(device=DEVICE, dtype=DTYPE)

train_params = list(filter(lambda p: p.requires_grad, policy.parameters())) + list(value_head.parameters())
optimizer = torch.optim.AdamW(train_params, lr=LR)

# ----------------------------
# 奖励模型（SequenceClassification + LoRA）
# ----------------------------
rm_base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE
)

# [FIX 3.2] 必须在加载 LoRA 适配器之前 resize
if added_pad_to_rm:
    rm_base.resize_token_embeddings(len(tok_rm))

# [FIX 3.3] 同样必须设置 pad_token_id
if tok_rm.pad_token_id is not None:
    rm_base.config.pad_token_id = tok_rm.pad_token_id

rm = PeftModel.from_pretrained(rm_base, RM_PATH).eval()
for p in rm.parameters(): p.requires_grad_(False)

# ----------------------------
# 任务数据
# ----------------------------
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

def format_prompt(p):
    return f"Human: {p}\nAssistant:"

# ----------------------------
# 生成回复
# ----------------------------
@torch.no_grad()
def generate_texts(model, prompts):
    inputs = tok_pol([format_prompt(p) for p in prompts], return_tensors="pt", padding=True)
    
    # [FIX 6.1] 手动将 inputs 移动到 model.device，防止 UserWarning
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        pad_token_id=tok_pol.pad_token_id,
        eos_token_id=tok_pol.eos_token_id
    )
    resp_lens = []
    for i in range(len(prompts)):
        resp_lens.append(int((gen_ids[i] != tok_pol.pad_token_id).sum().item()) - int((inputs["input_ids"][i] != tok_pol.pad_token_id).sum().item()))
    texts = tok_pol.batch_decode(gen_ids, skip_special_tokens=True)
    return gen_ids, resp_lens, texts

# ----------------------------
# 辅助函数
# ----------------------------
def last_hidden(model, input_ids, attention_mask):
    out = model.base_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1]
    idx = attention_mask.sum(dim=1) - 1
    idx = idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
    last_h = h.gather(1, idx).squeeze(1)
    return last_h

def response_logprobs(model, input_ids, attention_mask, resp_lens):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        ids = input_ids[:, 1:]
        lprobs = F.log_softmax(logits, dim=-1)
    lp_list = []
    B, Tm1, V = lprobs.shape
    for i, L in enumerate(resp_lens):
        if L <= 0:
            # [FIX 8.2] 确保这个 tensor 也在 policy 的设备上，而不是 'cuda:0'
            lp_list.append(torch.tensor(0.0, device=lprobs.device))
            continue
        start = (attention_mask[i].sum() - 1 - L).item()
        end   = start + L
        tok_ids = ids[i, start:end]
        tok_lps = lprobs[i, start:end, :].gather(-1, tok_ids.unsqueeze(-1)).squeeze(-1)
        lp_list.append(tok_lps.sum())
    return torch.stack(lp_list, dim=0)

@torch.no_grad()
def rm_reward(texts):
    enc = tok_rm(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # [FIX 6.2] 手动将 enc 移动到 rm.device
    enc = {k: v.to(rm.device) for k, v in enc.items()}

    y = rm(**enc).logits.view(-1)
    return y

def approx_kl(cur_lp, ref_lp):
    return (cur_lp - ref_lp).mean()

# ----------------------------
# 训练循环
# ----------------------------
policy.train()
value_head.train()

print(f"[INFO] device={DEVICE}, dtype={DTYPE}, params(trainable)={sum(p.numel() for p in train_params)/1e6:.2f}M")
for step in range(1, UPDATES + 1):
    batch_prompts = random.sample(PROMPTS, BATCH_SIZE)

    with torch.no_grad():
        gen_ids, resp_lens, texts = generate_texts(policy, batch_prompts)

    rm_texts = []
    for p, full in zip(batch_prompts, texts):
        rm_texts.append(full)

    r = rm_reward(rm_texts) # r 现在在 CPU 上

    attn = (gen_ids != tok_pol.pad_token_id).to(gen_ids.device).long()
    cur_lp = response_logprobs(policy, gen_ids, attn, resp_lens)
    ref_lp = response_logprobs(ref_model, gen_ids, attn, resp_lens)
    kl = approx_kl(cur_lp, ref_lp)

    with torch.no_grad():
        last_h = last_hidden(policy, gen_ids, attn)
    
    # [FIX 4.2] 显式将 last_h 移动到 value_head 所在的设备
    v = value_head(last_h.to(value_head.weight.device)).squeeze(-1) # v 现在在 CUDA 上

    # [FIX 6.3] r (CPU) 必须移动到 v (CUDA) 所在的设备
    adv = (r.to(v.device) - v).detach()
    
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    out = policy(input_ids=gen_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    ids    = gen_ids[:, 1:]
    lprobs = F.log_softmax(logits, dim=-1)

    b_token_lp = []
    for i, L in enumerate(resp_lens):
        if L <= 0:
            # [FIX 8.3] 确保这个 tensor 也在 policy 的设备上
            b_token_lp.append(torch.tensor(0.0, device=lprobs.device, requires_grad=True))
            continue
        start = (attn[i].sum() - 1 - L).item()
        end   = start + L
        tok_ids = ids[i, start:end]
        tok_lps = lprobs[i, start:end, :].gather(-1, tok_ids.unsqueeze(-1)).squeeze(-1)
        b_token_lp.append(tok_lps.sum())
    cur_lp_new = torch.stack(b_token_lp, dim=0)

    ratio = torch.exp(cur_lp_new - cur_lp.detach())
    clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    
    # [FIX 7.1] adv (on cuda:3) and ratio (on cuda:1) are on different devices.
    adv_on_device = adv.to(ratio.device)
    policy_loss = -(torch.min(ratio * adv_on_device, clipped * adv_on_device)).mean()


    with torch.set_grad_enabled(True):
        last_h2 = last_hidden(policy, gen_ids, attn)
        # [FIX 4.3] 再次显式移动
        v2 = value_head(last_h2.to(value_head.weight.device)).squeeze(-1)
    
    # [FIX 6.4] r (CPU) 必须移动到 v2 (CUDA) 所在的设备并转换 DTYPE
    value_loss = F.mse_loss(v2, r.to(v2.device, dtype=DTYPE))

    kl_loss = KL_COEF * kl

    # [FIX 8.1] value_loss (cuda:3) 必须被移动到 policy_loss (cuda:1) 所在的设备
    loss = policy_loss + VF_COEF * value_loss.to(policy_loss.device) + kl_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
    optimizer.step()

    if step % 5 == 0 or step == 1:
        print(f"[{step:04d}/{UPDATES}] loss={loss.item():.4f} | p_loss={policy_loss.item():.4f} | v_loss={value_loss.item():.4f} | kl={kl.item():.4f} | r={r.mean().item():.4f}")

policy.save_pretrained(OUT_DIR)
torch.save(value_head.state_dict(), os.path.join(OUT_DIR, "value_head.pt"))
print(f"[PPO] saved to {OUT_DIR}")