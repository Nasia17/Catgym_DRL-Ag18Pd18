# CatGym DRL Training

This repository includes standalone scripts for running CatGym training with both PPO and TRPO, extracted from `examples/notebooks/Train_CatGym.ipynb` while keeping the original scientific workflow intact.

## Scripts

- `examples/scripts/train_ppo.py`
  - Uses `agent=dict(type='ppo')`
  - Saves outputs to `./results/ppo_run`
- `examples/scripts/train_trpo.py`
  - Uses `agent=dict(type='trpo')`
  - Saves outputs to `./results/trpo_run`

Both scripts:
- avoid hard-coded system paths and use repository-relative imports,
- print episode reward and training progress every callback step,
- preserve environment setup, exploration configuration, parallel training, and agent-saving behavior.

## How to run

From the repository root:

```bash
python examples/scripts/train_ppo.py
```

```bash
python examples/scripts/train_trpo.py
```

## Notes

- Ensure dependencies from `requirements.txt` are installed in your environment.
- Training artifacts (plots, videos, rewards, and saved agent checkpoints) are written under `./results/ppo_run` and `./results/trpo_run`.
