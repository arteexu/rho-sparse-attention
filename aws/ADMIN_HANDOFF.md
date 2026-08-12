# Admin handoff for the 12-hour RHO block run

I want to run one bounded RHO/SLM experiment on the edu-llm capacity block.

Planned run:

- run name: `rho-tinyllama-12h-01`
- repository: `<owner>/<repo>`
- branch: `<your-public-branch>`
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

Approval needed:

1. Confirm that a capacity-block fleet is currently up, or launch one if it is not.
2. Confirm I can claim one idle `p5.48xlarge` node for this run.
3. Confirm the branch/repo is acceptable for the block lane, which clones public GitHub repos.
4. Optional for the normal citable platform path: register this repo in `edu-llm/platform`,
   create its ECR repository/grants, deploy the publisher role update, and set
   `AWS_ECR_PUBLISHER_ROLE_ARN` as an Actions repository variable. The block run does not
   require this image path.

I will dry-run the distributed workflow first, start only if the plan looks correct, monitor W&B
and `block-logs.yml`, and release the node with `block-release.yml` after the run exits.
