# Admin handoff for the 12-hour RHO block run

I want to run one bounded RHO/SLM experiment on the edu-llm capacity block.

Planned run:

- run name: `rho-tinyllama-12h-01`
- repository: `arteexu/rho-sparse-attention`
- branch: `edullm/rho-block-12h`
- commit: `216b9fbb4b156ac01db23f95396826e24e0a4c91`
- workflow: `Block: start one run across several nodes`
- node count: `1`
- command: `bash .edullm/block_rho_12h_suite.sh`
- mesh flags: `false`
- fabric: `auto`
- region: `us-east-2`
- W&B project: `capacity-block`
- max wall-clock target: under 12 hours

What it runs:

- CLM baseline
- random masking baseline
- RHO-style SLM with curriculum `0:0.6,300:0.8,650:1.0`
- model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- reference model: `TinyLlama/TinyLlama_v1.1_math_code`
- corpus: `open-web-math/open-web-math`

Timing note:

- This suite is configured for a 12-hour budget on a fresh block: three strategies, each
  capped at 3.5 hours, plus setup/download slack.
- The active August 2026 block guide says the current block is usable only until
  2026-08-12 11:00 UTC. At 2026-08-12 04:55 UTC, that leaves about 6 hours, so a full
  12-hour proof run needs the next block or a reduced command.
- If we must use the remainder of the current block, use a smaller command such as:
  `env STRATEGIES=clm,slm MAX_STEPS=450 MAX_RUNTIME_HOURS_PER_STRATEGY=2.75 bash .edullm/block_rho_12h_suite.sh`

Approval needed:

1. Confirm that a capacity-block fleet is currently up, or launch one if it is not.
2. Confirm I can claim one idle `p5.48xlarge` node for this run.
3. Confirm the branch/repo is acceptable for the block lane, which clones public GitHub repos.
4. Optional for the normal citable platform image path: create or mirror this repository under
   the `edu-llm` organization, then register it in `edu-llm/platform`, create its ECR
   repository/grants, deploy the publisher role update, and set `AWS_ECR_PUBLISHER_ROLE_ARN`
   as an Actions repository variable. The current publisher role trust is scoped to the
   `edu-llm` organization, so the personal `arteexu/*` repo can run on the block but cannot
   publish a platform image by variable alone.

Current setup status:

- `edullm` CLI is installed locally: `edullm 4.5.0 (bc007e5506f8)`.
- The repository is public and pushed to GitHub.
- The block entrypoint and `.edullm/run.yaml` are present.
- The research-image workflow exists, but its first run failed because
  `AWS_ECR_PUBLISHER_ROLE_ARN` is not set and the repo is not yet registered for the platform
  image lane.

I will dry-run the distributed workflow first, start only if the plan looks correct, monitor W&B
and `block-logs.yml`, and release the node with `block-release.yml` after the run exits.
