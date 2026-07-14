---
name: review-fix-loop
description: Run a bounded, cost-controlled, security-gated review-fix-reverify loop on code changes using the /code-review and /security-review slash commands. Use before a commit/PR, or when asked to review and remediate code across multiple passes — not for a single one-off review (use /code-review directly for that).
---

# Review-Fix Loop (Claude Code)

Claude-Code-native counterpart to `.agents/skills/review-fix-loop/SKILL.md` (that one is Antigravity's copy — same procedure, different execution primitives). This version drives Claude Code's own `/code-review` and `/security-review` slash commands rather than a generic manual review pass.

## Workflow

### Step 1 — Scope Gate
Before reviewing, check whether the diff touches auth, uploads, secrets/tokens, or external/untrusted input. If yes, Step 5 (Security Gate) is **mandatory**, not optional.

### Step 2 — Review Pass 1
Invoke `/code-review` at **`medium`** effort by default — not `high`/`ultra`. Escalate effort only for large or high-stakes diffs; `ultra` (multi-agent cloud review) is reserved for pre-release changes, never the default per-pass effort. Record every finding with file:line, severity, and a concrete failure scenario. Classify each as `CONFIRMED` or `PLAUSIBLE`.

### Step 3 — Fix
Apply fixes for `CONFIRMED` findings. Flag `PLAUSIBLE` findings to the user rather than silently patching them.

### Step 4 — Bounded Re-Review (max 2 more passes, early-exit)
Re-run `/code-review` after each fix.
- Zero findings on a pass → stop immediately, don't spend a further pass confirming it's clean.
- Findings still remain after 3 total passes → stop anyway and hand the residue to the user. Never loop unbounded.

### Step 5 — Security Gate (once, at the end)
Run `/security-review` on the **final** diff only, not per-iteration — cheaper, and catches a different issue class (secrets, injection) than general review.

### Step 6 — Human Checkpoint
Present the consolidated finding list (tagged `fixed`/`skipped`/`no_change_needed`), the final diff, and an explicit approval ask before any commit/push. Never auto-commit — this mirrors the standing rule Claude Code already operates under for hard-to-reverse actions.

## Cost/risk levers (don't skip these)
| Lever | Default |
|---|---|
| Review effort | `medium`, escalate only when warranted |
| Iteration cap | 3 total passes, early-exit on first clean pass |
| Security gate placement | Once, after the loop, on the final diff |

Full rationale and cross-tool (Antigravity / GitHub Copilot) comparison: [Reference/ai_code_review_workflow.md](../../Reference/ai_code_review_workflow.md).
