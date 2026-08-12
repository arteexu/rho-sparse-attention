# 12-hour RHO run on the edu-llm capacity block

Use the GitHub Actions workflows whose names start with `Block:` in
`edu-llm/platform`.

## 1. Check Capacity

Dispatch `Block: which node is free` from `main`.

With GitHub CLI:

```bash
gh workflow run block-status.yml --ref main -R edu-llm/platform \
  -f region=us-east-2
```

Pick a node that reports `IDLE`.

## 2. Recommended First Launch

Use `Block: start one run across several nodes`, even for one node. It supports
`dry_run` and composes the `torchrun` launcher for the node.

Actual repository settings for this run:

- repository: `arteexu/rho-sparse-attention`
- branch: `edullm/rho-block-12h`
- command: `bash .edullm/block_rho_12h_suite.sh`

Dry run:

```bash
gh workflow run block-run-distributed.yml --ref main -R edu-llm/platform \
  -f run_name=rho-tinyllama-12h-01 \
  -f branch=edullm/rho-block-12h \
  -f repository=arteexu/rho-sparse-attention \
  -f command='bash .edullm/block_rho_12h_suite.sh' \
  -f node_count=1 \
  -f nodes= \
  -f expert_parallel= \
  -f mesh_flags=false \
  -f wandb_project=capacity-block \
  -f fabric=auto \
  -f dry_run=true \
  -f region=us-east-2
```

If the plan looks right, run the same dispatch with `dry_run=false`.

```bash
gh workflow run block-run-distributed.yml --ref main -R edu-llm/platform \
  -f run_name=rho-tinyllama-12h-01 \
  -f branch=edullm/rho-block-12h \
  -f repository=arteexu/rho-sparse-attention \
  -f command='bash .edullm/block_rho_12h_suite.sh' \
  -f node_count=1 \
  -f nodes= \
  -f expert_parallel= \
  -f mesh_flags=false \
  -f wandb_project=capacity-block \
  -f fabric=auto \
  -f dry_run=false \
  -f region=us-east-2
```

This bounded suite runs:

- `clm`
- `random`
- `slm`

Each strategy defaults to at most 3.5 hours, so the whole suite stays under a
12-hour wall clock including setup/download time.

If the active block has less than 12 hours left, do not use the full command above.
Use a reduced command instead, for example:

```bash
gh workflow run block-run-distributed.yml --ref main -R edu-llm/platform \
  -f run_name=rho-tinyllama-short-01 \
  -f branch=edullm/rho-block-12h \
  -f repository=arteexu/rho-sparse-attention \
  -f command='env STRATEGIES=clm,slm MAX_STEPS=450 MAX_RUNTIME_HOURS_PER_STRATEGY=2.75 bash .edullm/block_rho_12h_suite.sh' \
  -f node_count=1 \
  -f nodes= \
  -f expert_parallel= \
  -f mesh_flags=false \
  -f wandb_project=capacity-block \
  -f fabric=auto \
  -f dry_run=true \
  -f region=us-east-2
```

## 3. What The Command Runs

Defaults in `.edullm/block_rho_12h_suite.sh`:

- model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- reference: `TinyLlama/TinyLlama_v1.1_math_code`
- dataset: `open-web-math/open-web-math`
- sequence length: 2048
- global tokens per optimizer step on one 8-GPU node: about 131k
- max steps per strategy: 900
- SLM curriculum: `0:0.6,300:0.8,650:1.0`

Override any setting by prefixing environment variables inside the `command`
field, for example:

```text
env MAX_STEPS=1200 MAX_RUNTIME_HOURS_PER_STRATEGY=3.75 bash .edullm/block_rho_12h_suite.sh
```

## 4. Watch Logs

The workflow summary prints the elected node and the W&B link. To read logs:

```bash
gh workflow run block-logs.yml --ref main -R edu-llm/platform \
  -f node=<elected-node> \
  -f run_name=rho-tinyllama-12h-01 \
  -f lines=300 \
  -f region=us-east-2
```

Small artifacts are mirrored under the synced log prefix:

```text
log/artifacts/clm/train_log.csv
log/artifacts/random/train_log.csv
log/artifacts/slm/train_log.csv
```

Full local checkpoints are under:

```text
/work/runs/<run_name>/<strategy>/checkpoint-last/
```

Those are also under the node's `/scratch/<run_name>/repo` tree and are copied
by the capacity-block drain/scratch sync.

## 5. Release The Node

After the run exits, give the claim back:

```bash
gh workflow run block-release.yml --ref main -R edu-llm/platform \
  -f nodes=<elected-node> \
  -f region=us-east-2
```

The release workflow refuses if the container is still running.
