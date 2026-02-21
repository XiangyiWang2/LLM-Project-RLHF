import os, math, random, torch, contextlib
from torch.cuda.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

DEVICE_MAP = "auto"
DTYPE = torch.float16

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SFT_PATH = "checkpoints/sft-tinyllama"    
RM_PATH  = "checkpoints/reward-model"  
OUT_DIR  = "checkpoints/ppo-tinyllama"   
os.makedirs(OUT_DIR, exist_ok=True)


tok = AutoTokenizer.from_pretrained(RM_PATH, use_fast=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
tok.padding_side = "left"
vocab_size = len(tok)


ref_base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=DTYPE, device_map=DEVICE_MAP
)
ref = PeftModel.from_pretrained(ref_base, SFT_PATH)
ref.eval()


pol_base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=DTYPE, device_map=DEVICE_MAP
)
policy = PeftModel.from_pretrained(pol_base, SFT_PATH)
policy.train()

def ensure_resize(model):
    base = getattr(model, "base_model", None) or getattr(model, "model", None) or model
    if hasattr(base, "get_input_embeddings"):
        emb = base.get_input_embeddings()
        if emb is not None and emb.num_embeddings != vocab_size:
            base.resize_token_embeddings(vocab_size)
    if hasattr(model, "config"): model.config.pad_token_id = tok.pad_token_id
    if hasattr(base, "config"):  base.config.pad_token_id = tok.pad_token_id

ensure_resize(ref); ensure_resize(policy)


rm_base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, torch_dtype=DTYPE, device_map=DEVICE_MAP
)
ensure_resize(rm_base)
rm = PeftModel.from_pretrained(rm_base, RM_PATH)
rm.eval()


raw = load_dataset("stanfordnlp/SHP", split="train[:1%]")
prompts = [ex.get("history","") for ex in raw if isinstance(ex.get("history",""), str) and len(ex["history"])>0]
random.seed(42); random.shuffle(prompts)
prompts = prompts[:512]


BATCH = 8              
STEPS = math.ceil(len(prompts)/BATCH)
MAX_NEW = 128
TOP_P = 0.9
TEMP  = 1.0
BETA  = 0.02            
LR    = 1e-5

optim = torch.optim.AdamW(policy.parameters(), lr=LR)

def model_device(model):
    return next(model.parameters()).device

def sample_response(model, batch_prompts):
    device = model_device(model)
    inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True, top_p=TOP_P, temperature=TEMP, pad_token_id=tok.pad_token_id)
    resp = out[:, inputs["input_ids"].shape[1]:]  # 仅响应部分
    return inputs["input_ids"], resp

def seq_logprob(model, full_ids, resp_len, need_grad):
    # 计算“响应”区间的句级 logprob（对当前策略需保留计算图）
    if torch.cuda.is_available():
        ctx = torch.amp.autocast('cuda', dtype=DTYPE) if need_grad else torch.no_grad()
    else:
        ctx = contextlib.nullcontext() if need_grad else torch.no_grad()

    with ctx:
        logits = model(full_ids[:, :-1]).logits          
        targets = full_ids[:, 1:]                         
        logits_resp  = logits[:, -resp_len:, :]          
        targets_resp = targets[:, -resp_len:]
        logprobs = torch.log_softmax(logits_resp, dim=-1)
        token_lp = logprobs.gather(-1, targets_resp.unsqueeze(-1)).squeeze(-1)  
        seq_lp   = token_lp.sum(dim=-1)                   
    return seq_lp

def rm_score(texts):
    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model_device(rm))
    with torch.no_grad():
        s = rm(**inputs).logits.squeeze(-1).float().detach()
    return s.to(model_device(policy))


for step in range(STEPS):
    batch = prompts[step*BATCH:(step+1)*BATCH]
    if not batch: break


    prompt_ids, resp_ids = sample_response(policy, batch)
    full_ids = torch.cat([prompt_ids.to(model_device(policy)), resp_ids.to(model_device(policy))], dim=1)
    resp_len = resp_ids.shape[1]


    lp_cur = seq_logprob(policy, full_ids, resp_len, need_grad=True) 
    lp_ref = seq_logprob(ref,    full_ids, resp_len, need_grad=False)  


    kl  = (lp_cur - lp_ref)               
    texts = [f"Human: {p}\nAssistant: {r}" for p, r in zip(batch, tok.batch_decode(resp_ids, skip_special_tokens=True))]
    rew = rm_score(texts)


    advantage = (rew - BETA * kl).detach()     
    loss = -(advantage * lp_cur).mean()

    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optim.step()

    if (step+1) % 5 == 0 or step == 0:
        print(f"[RL] step {step+1}/{STEPS} | loss={loss.item():.4f} | rew={rew.mean().item():.3f} | kl={kl.mean().item():.3f}")


policy.eval()
policy.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"[RL] saved to {OUT_DIR}")
