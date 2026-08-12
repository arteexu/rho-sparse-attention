# AWS RHO-1 Scale Test

This folder contains launch helpers for a paper-adjacent RHO-1 run on AWS.

The recommended first AWS test is a 1.1B TinyLlama continual-pretraining run on
OpenWebMath using RHO-style selective loss. OpenWebMath contains 6.3M documents
and about 14.7B tokens of mathematical web text, which makes it the closest
open dataset target to the RHO-1 math setup in this repo. The default model is
`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, and the default reference
model is `TinyLlama/TinyLlama_v1.1_math_code`.

For hardware, start with one `p4d.24xlarge` or `p5.48xlarge`. AWS documents P4d
as A100-based and P5 as H100/H200-family GPU instances; the Deep Learning AMI
docs list P4 as up to 8 A100 GPUs and P5 as up to 8 H100 GPUs. P5 is preferred
if available, while P4d is a reasonable A100 baseline.

## Instance Setup

Use a recent AWS Deep Learning AMI with CUDA/PyTorch. Then:

```bash
git clone <your-repo-url> rho-sparse-attention
cd rho-sparse-attention
bash aws/setup_instance.sh
```

If the repo is already on the instance:

```bash
cd "/path/to/Sparse Attention"
bash aws/setup_instance.sh
```

Optional but recommended:

```bash
huggingface-cli login
aws configure
```

## 1B RHO-Style Run

```bash
bash aws/run_tinyllama_rho_1b.sh
```

This launches:

```bash
torchrun --standalone --nproc_per_node=8 rho_hf_pretrain.py ...
```

Outputs go to:

```text
runs/aws_tinyllama_rho_1b/
```

Resume with the same command; the script uses `--resume` and
`checkpoint-last/`.

## Baselines

Run the CLM baseline:

```bash
STRATEGY=clm OUTPUT_DIR=runs/aws_tinyllama_clm_1b bash aws/run_tinyllama_rho_1b.sh
```

Run random masking at the same schedule:

```bash
STRATEGY=random OUTPUT_DIR=runs/aws_tinyllama_random_1b bash aws/run_tinyllama_rho_1b.sh
```

The headline comparison should be:

- selected tokens to reach CLM eval-loss target
- total tokens seen to reach CLM eval-loss target
- wall-clock time to reach CLM eval-loss target
- final eval loss at matched token budget

## Notes

- This is closer to the RHO-1 paper than the local custom MoE run because it uses
  a 1.1B pretrained model and OpenWebMath.
- It is still not a full paper reproduction: the paper trains reference models
  on larger curated corpora and runs much larger token budgets.
- The script computes reference losses online. For a stronger wall-clock claim,
  add offline reference-score precomputation as the next step.

## edu-llm Capacity Block

If you are using the GitHub Actions capacity-block workflows in
`edu-llm/platform`, use the 12-hour block runbook instead:

```text
aws/BLOCK_12H.md
```

That path uses `.edullm/block_rho_12h_suite.sh` and runs CLM, random masking,
and RHO-style SLM in one bounded job.
