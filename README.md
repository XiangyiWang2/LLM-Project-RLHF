# Mini-RLHF (TinyLlama)
Pipeline: SFT -> Reward Model (pairwise approximation) -> PPO (KL-penalty).
Run order:
  1) python scripts/01_train_sft.py
  2) python scripts/02_train_rm.py
  3) python scripts/03_ppo_rlhf.py
