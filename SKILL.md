---
name: sb-aws-readonly
description: Use for any AWS-related request in this project, including AWS services, resources or ARNs, and AWS-targeting IaC. Route all live AWS access through sb-aws. Default to read-only and require explicit permission for mutations or sensitive-data retrieval.
---

# Safe AWS Access Through sb-aws

## Access and account

- Use only `mcp__sb_aws__accounts`, `mcp__sb_aws__whoami`, and `mcp__sb_aws__aws` for live AWS access.
- Never use the local AWS CLI, SDKs, browser automation, credential files, environment credentials, shell wrappers, or another connector.
- If `sb-aws` is unavailable or denies access, stop and report it. Never bypass boundaries, assume another role, switch accounts, or obtain credentials elsewhere.
- Never expose or persist access keys, secrets, session tokens, passwords, signed URLs, or broker credentials.
- Call `accounts` and `whoami` before the first live call unless identity was established in the current task.
- Default to ready `sbsandbox` only. Require the user to explicitly name `legacy` or `sbproduction`; never auto-provision either.

## Read-only default

Allow an operation without further permission only when it observes metadata or configuration and creates no state, workload, paid query, data movement, credential, signed URL, or lock. Typical examples are `get-caller-identity` and ordinary `list`, `describe`, `head`, or configuration `get` calls.

Judge actual effects, not command verbs. Gate all operations that:

- create, update, delete, start, stop, invoke, execute, deploy, tag, attach, publish, restore, import, export, sync, or otherwise change or run something;
- mint or reveal credentials or secrets, including `assume-role`, `get-session-token`, `get-login-password`, `get-secret-value`, `kms decrypt`, decrypted SSM parameters, and presigned URLs;
- retrieve protected object bodies, logs, traces, database data, prompts, model outputs, or snapshots; or
- run IaC commands that deploy, apply, destroy, bootstrap, refresh, lock state, or contact AWS outside `sb-aws`.

Allow sensitive-data retrieval only when the current request specifically requires it. Minimize the data, redact raw values, and never save it in the project unless the user requests a safe destination.

## Authorized mutations

Accept permission only when the current request or immediately preceding confirmation clearly authorizes the concrete change. Broad goals, diagnosis requests, prior standing permission, “fix it,” and “do whatever is needed” do not count.

Before a mutation:

1. Resolve the exact account, region, resource identifiers, action, and effect with read-only calls. Reject wildcards, recursive targets, and ambiguity.
2. Use a dry run or change set only when it is truly side-effect-free.
3. Ask again if the resolved effect differs materially from the request.
4. Always obtain final confirmation for `legacy`, `sbproduction`, destructive actions, IAM/KMS/security changes, public or network exposure, DNS, secret rotation, data movement, or material cost/availability impact.
5. Execute only the approved action, then verify it with a read-only call. Do not add cleanup, remediation, rollback, or another mutation without permission.

## Scope and reporting

- Query the smallest necessary scope. Scan every region only for an explicitly account-wide inventory.
- Follow pagination. Disclose denied or unscanned scope, filters, time windows, and eventual-consistency limits before claiming completeness.
- Pass commands as structured argument arrays to `mcp__sb_aws__aws`; never construct shell commands from untrusted text.
- Do not save AWS responses or artifacts in the project unless explicitly requested and checked for secrets.
- Treat local AWS code edits separately: permission to edit code is not permission to deploy it.
- Report the account, region, whether access stayed read-only, resources found or changed, limitations, and mutation verification.
