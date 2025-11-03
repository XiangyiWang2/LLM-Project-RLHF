# Mini-RLHF (TinyLlama)

Pipeline: **SFT → Reward Model → PPO (KL-penalty)**

## Run
1) `pip install -r requirements.txt`
2) `python scripts/01_train_sft.py`        # outputs: checkpoints/sft-tinyllama
3) `python scripts/02_train_rm.py`         # outputs: checkpoints/reward-model
4) `python scripts/03_ppo_rlhf.py`         # outputs: checkpoints/ppo-tinyllama
5) `python scripts/04_eval.py`             # compare SFT vs PPO (RM win-rate, CSV)
