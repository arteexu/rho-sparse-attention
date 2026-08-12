# RHO-1 + MoE Mini Experiment

This workspace contains a small, runnable experiment inspired by the paper
`Not All Tokens Are What You Need`.

The goal is not to reproduce the paper's scale. It is to capture the key
mechanics in a setting that can run on a laptop:

- train a reference language model on high-quality desired data
- score pretraining tokens with reference-model loss
- train with normal causal language modeling (CLM)
- train with Selective Language Modeling (SLM), masking loss to high-score tokens
- compare dense transformer and Mixture-of-Experts (MoE) transformer variants
- report desired-domain validation loss/accuracy, noisy-domain loss, token selection
  diagnostics, and MoE routing/load statistics

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

If `uv` has not installed Python 3.12 yet, run:

```bash
uv python install 3.12
```

## Quick Smoke Run

```bash
python rho_moe_experiment.py --preset smoke --device auto
```

This runs four small experiments:

- dense CLM baseline
- dense SLM
- MoE CLM baseline
- MoE SLM

Outputs are written to `runs/rho_moe_<timestamp>/`.

## More Useful Local Run

```bash
python rho_moe_experiment.py \
  --preset small \
  --device auto \
  --steps 500 \
  --ref-steps 250 \
  --select-ratio 0.6
```

## What To Look For

The main signal is whether SLM improves loss/accuracy on `desired_eval` relative
to CLM while ignoring a meaningful fraction of tokens. In the synthetic corpus,
desired tokens are math-like reasoning and operator tokens, while noisy tokens
include metadata, IDs, markup, and random strings. This lets you inspect whether
SLM focuses gradients on useful content without deleting the full context.

For MoE runs, inspect:

- `router_entropy`
- `expert_fraction_*`
- `load_balance_loss`

These show whether the router collapses to a single expert or spreads traffic
across experts.

The default MoE uses top-2 routing over 4 experts. You can change this with:

```bash
python rho_moe_experiment.py --preset small --moe-top-k 1
```

## Real-Data Run

`rho_moe_real_train.py` uses real Hugging Face datasets instead of the synthetic
corpus. By default it trains:

- reference model: GSM8K question+answer text
- pretraining mixture: WikiText-2 plus a fraction of GSM8K text
- desired eval: GSM8K test text
- general eval: WikiText-2 validation text

Quick smoke run:

```bash
python rho_moe_real_train.py --preset smoke --run all --device auto
```

Small local run:

```bash
python rho_moe_real_train.py --preset small --run all --device auto
```

The default real run uses a local word-level tokenizer trained from the loaded
text so the tiny model learns quickly. To use a Hugging Face tokenizer instead:

```bash
python rho_moe_real_train.py \
  --preset small \
  --tokenizer-name gpt2 \
  --run moe \
  --device auto
```

Useful real-run columns:

- `desired_loss`, `desired_acc`: desired-domain eval performance
- `general_loss`, `general_acc`: general-corpus eval performance
- `selected_fraction`: fraction of target positions used for SLM loss
- `selected_desired_source_fraction`: how much SLM selected from desired-domain
  examples inside the pretraining mixture
- `expert_fraction_*`: MoE routing balance

## Efficiency Sweeps

`rho_efficiency_sweep.py` runs multiple seeds and aggregates the curves. This is
the recommended entry point for testing the efficiency claim.

Dense selection-objective sweep:

```bash
python rho_efficiency_sweep.py \
  --seeds 11,12,13 \
  --preset small \
  --run efficiency_dense \
  --device auto \
  -- --steps 80 --ref-steps 40 --eval-every 20
```

MoE sweep:

```bash
python rho_efficiency_sweep.py \
  --seeds 11,12,13 \
  --preset small \
  --run efficiency_moe \
  --device auto \
  -- --steps 80 --ref-steps 40 --eval-every 20
```

The sweep writes:

- `aggregate_final.csv`: mean/std final metrics by method
- `curves.csv`: all learning-curve points
- `target_crossings.csv`: whether each method reached the same final
  desired-domain loss as its CLM baseline
- `aggregate_target_crossings.csv`: hit rate and token/time cost to reach that
  CLM target

The strongest efficiency evidence is not just a lower final loss. Look for SLM
reaching the CLM target with fewer `hit_selected_tokens`, fewer
`hit_total_tokens`, or lower `hit_elapsed_sec`, while random masking at the same
ratio does not.

## Larger Local Runs

Two larger presets are available:

- `medium`: larger than the quick sweeps, usually suitable for iteration.
- `large_local`: a much larger custom MoE transformer and more packed real-text
  sequences. This is the closest local version of the paper-style continual
  pretraining experiment in this repo, but it is still far smaller than RHO-1's
  1B/7B runs.

Recommended long MoE efficiency run with an SLM-to-CLM curriculum:

```bash
python rho_moe_real_train.py \
  --preset large_local \
  --run efficiency_moe \
  --device auto \
  --resume-dir runs/large_local_moe_seed11 \
  --select-ratio-schedule 0:0.6,800:0.8,1600:1.0
```

Resume the same run with the exact same command. The trainer writes:

- `reference_checkpoint.pt`
- `checkpoints/<run_name>.pt`
- `<run_name>.csv`
- `summary.csv`

For a smaller rehearsal before committing to the long run:

```bash
python rho_moe_real_train.py \
  --preset medium \
  --run efficiency_moe \
  --device auto \
  --resume-dir runs/medium_moe_seed11 \
  --select-ratio-schedule 0:0.6,300:0.8,600:1.0
```

For multi-seed larger sweeps, start with `medium`:

```bash
python rho_efficiency_sweep.py \
  --seeds 11,12,13 \
  --preset medium \
  --run efficiency_moe \
  --device auto \
  -- --select-ratio-schedule 0:0.6,300:0.8,600:1.0
```
