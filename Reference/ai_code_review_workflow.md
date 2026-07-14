# AI-Assisted Code Review Workflow — "Pre-Commit Remediation Loop" + "TDD Planning Gate"

A repeatable process for using a coding agent to review, fix, and re-verify code changes — with explicit cost, security, and optimization guardrails. Rather than inventing a new pattern, this composes two things already distilled in [Vibe Engineering](Vibe%20Engineering/vibeEngineeringAssignments.md) Day 4 (Secure AI Code codelab, entry 8), plus the Day 4 Ambient Expense Agent's bounded eval/grade loop:

- **TDD Planning Gate (Test Driven Development)** — the codelab's own step: append a planning-gate rule to `CONTEXT.md` requiring a dedicated security section in every implementation plan; the agent presents the plan for approval *before* generating code. This is the "scope gate" below, moved to the front of the loop.
- **Pre-Commit Remediation Loop** — this document's name for the mechanism the codelab demonstrates at its pre-commit-gating step: a deterministic hook (Semgrep) blocks a commit on a real finding (a hardcoded API key) → the agent automatically refactors → tests are re-run to confirm they're still green → the commit is retried. It's a bounded loop because it's anchored to a hook's pass/fail exit code, not an open-ended "keep reviewing" instruction.

This document generalizes that pair into a tool-agnostic loop, then maps it concretely onto **Claude Code**, **Antigravity**, and **GitHub Copilot** — since the three implement the same idea with different native primitives.

---

## 1. The Loop

```
Scope Gate → Review Pass 1 → Fix → Re-Review (bounded, early-exit) → Security Gate (once) → Human Checkpoint
```

### Step 1 — Scope Gate
Before reviewing, determine whether the diff touches auth, file/image uploads, secrets, or external input (e.g. `api/main.py`'s `/search` upload handling). If yes, a Security Gate pass (Step 5) is **mandatory**, not optional.

### Step 2 — Review Pass 1
Run the review at **`medium`** effort by default — not `high` or `ultra`. Effort level directly drives cost:
- `low`/`medium`: fewer, high-confidence findings. Cheap. Good default.
- `high`→`max`: broader coverage, more uncertain findings included. Use when the diff is large or high-stakes.
- `ultra`: multi-agent cloud review. Reserve for pre-release or high-stakes changes only — never the default per-iteration effort.

List every finding with file:line, severity, and a concrete failure scenario before touching any code.

### Step 3 — Fix
Apply fixes for `CONFIRMED` findings automatically. Flag `PLAUSIBLE` findings for human judgment rather than silently patching them — mirrors the human-in-the-loop threshold in the Day 4 ambient expense agent (auto-handle clear-cut cases, escalate ambiguous ones instead of guessing).

### Step 4 — Bounded Re-Review (max 2 more passes, early-exit)
Re-run the review after each fix.
- If a pass returns **zero findings, stop immediately** — don't spend a further pass proving nothing's left.
- If findings remain after **3 total passes**, stop anyway and hand the residue to a human instead of looping indefinitely.

This bound is the real cost control. Unbounded "keep checking" loops are where agentic review spend runs away — an early-exit-on-clean-pass rule plus a hard iteration cap prevents that.

### Step 5 — Security Gate (once, at the end)
Run a dedicated security review on the **final** diff only — not per-iteration. It catches a different class of issue (secrets, injection, STRIDE-style threats per the Day 4 pattern) than a general code review, and running it once at the end is far cheaper than folding it into every loop pass.

### Step 6 — Human Checkpoint (never auto-commit)
Present:
1. The consolidated finding list across all passes, tagged `fixed` / `skipped` / `no_change_needed`.
2. The final diff.
3. An explicit ask for approval before any commit/push.

Never skip this step — it's the same "pause before the hard-to-reverse action" principle as the harness-design guidance from Day 1.

---

## 2. The Three Levers That Actually Control Cost & Risk

| Lever | Default | Why |
|---|---|---|
| **Review effort level** | `medium`, not `ultra` | Effort level is the primary cost driver; escalate only when a diff is large/high-stakes. |
| **Iteration cap** | 3 total passes, early-exit on first clean pass | Prevents unbounded review loops from silently running up cost. |
| **Security gate placement** | Once, after the loop, on the final diff | Cheaper than per-iteration; catches a distinct issue class (secrets/injection) that general review may deprioritize. |

Skipping any of the three either wastes budget (uncapped loops, always-`ultra`) or creates risk (no security gate, auto-committing without a human checkpoint).

---

## 3. Claude Code Implementation

> This maps almost exactly onto a pattern you've already catalogued: the "Pre-Commit Remediation Loop" + "TDD Planning Gate" from the Day 4 Secure AI Code codelab, combined with the bounded eval/grade loop from the Day 4 ambient expense agent. Rather than inventing a new pattern, compose what's already available:
>
> **The loop, with guardrails baked in:**
>
> 1. **Scope gate** — before reviewing, flag whether the diff touches auth/uploads/secrets/external input (like `api/main.py`'s upload handling). If yes, a `/security-review` pass is mandatory later, not optional.
> 2. **Review pass 1** — run `/code-review` at `medium` effort by default, not `high`/`ultra`. Cost scales with effort level; `ultra` spins up a multi-agent cloud review and should be reserved for pre-release/high-stakes diffs, not every loop iteration. List every finding (file:line, severity, failure scenario) before touching code.
> 3. **Fix** — apply fixes for `CONFIRMED` findings automatically; `PLAUSIBLE` ones get flagged, not silently patched — this is the same human-in-the-loop threshold logic as the ambient expense agent (auto-handle the clear-cut cases, escalate the ambiguous ones instead of guessing).
> 4. **Bounded re-review** — max 2 more passes, with early exit. Re-run `/code-review` after each fix. If a pass returns zero findings, stop immediately — don't burn a third pass proving nothing's left. If findings still remain after 3 total passes, stop anyway and hand the residue to a human rather than looping indefinitely. This bound is the actual cost control — unbounded "keep checking" loops are where agentic review spend runs away.
> 5. **Security gate, once, at the end** — run `/security-review` on the final diff, not per-iteration. It catches a different class of issue (secrets, injection, STRIDE-style) than generic code-review, and running it once at the end is cheap relative to folding it into every pass.
> 6. **Human checkpoint** — never auto-commit. Present the consolidated finding list (tagged `fixed`/`skipped`/`no_change_needed`), the final diff, and an explicit ask before any commit/push. This isn't optional — it's already a hard rule in how the agent operates, and it's the same "pause before the hard-to-reverse action" principle from Day 1's harness-design material.
>
> The three levers that actually control cost/security here are: effort level defaults to `medium` not `ultra`, iteration cap of 3 with early-exit on a clean pass, and security review runs once at the end, not per-loop. Skipping any of the three either wastes budget (uncapped loops, always-`ultra`) or creates risk (no security gate, auto-committing).

**Concrete mechanisms this relies on:**
- `/code-review` and `/security-review` are pre-built slash-command skills — the effort-level system (`low`/`medium`/`high`→`max`/`ultra`) is a real, invokable parameter, not a convention to remember.
- The "never auto-commit" rule isn't project config — it's a standing instruction the agent already operates under (confirm before any commit/push, force-push, or destructive git operation).
- **Packaged Skill**: `.claude/skills/review-fix-loop/SKILL.md` — a Claude-Code-native, project-scoped skill that drives `/code-review` + `/security-review` directly. This is a separate file from `.agents/skills/review-fix-loop/SKILL.md` (Antigravity's copy, driving Semgrep/manual review instead) — they cover the same procedure but live in the two locations each tool actually auto-discovers.
- **Important caveat**: this project's `.agents/skills/` directory (where `vibe-engineering-advisor`, `disclaimer-manager`, `stride-threat-model`, `review-fix-loop`, etc. live) is Antigravity's convention, **not** one of Claude Code's own auto-discovered skill locations — none of those files appear in Claude Code's live "available skills" list. Claude Code instead auto-discovers `.claude/skills/` (project-scoped, used here) and `~/.claude/skills/` (global, cross-project — not used here; scoped to this project only per an explicit choice when asked). `.agents/skills/*` remain readable/applicable manually in Claude Code, and become genuinely dynamic, self-triggering skills only when the same project is opened in Antigravity (see below).

---

## 4. Antigravity Implementation

Antigravity is where this pattern *actually originates* — the Day 4 Secure AI Code codelab is an Antigravity walkthrough, and this project's `.agents/skills/` / `.agents/CONTEXT.md` / `.agents/hooks.json` layout is Antigravity's own convention, not something borrowed from elsewhere. Mapped step-by-step:

1. **Scope gate → `.agents/CONTEXT.md` planning-gate rule.** Exactly the codelab's step: a rule requiring a dedicated security section in every implementation plan, enforced because Antigravity presents plans for approval *before* generating code — the gate is structural, not something the agent has to remember to do.
2. **Review pass → Semgrep + conversational review.** Antigravity doesn't ship a built-in `/code-review` slash command with named effort levels the way this session does. The equivalent is a `.semgrep/rules.yaml` static-analysis pass (as in the codelab) plus asking the agent directly — effort-level discipline (start cheap, escalate only when needed) has to be self-imposed via how the request is framed, not selected as a parameter.
3. **Fix → same, gated by tool-permission mode.** Antigravity CLI's five permission modes (`request-review`, `proceed-in-sandbox`, `always-proceed`, `strict`) directly control how much autonomy the fix step gets — `request-review` mode makes "flag `PLAUSIBLE` findings instead of auto-patching" the *default* behavior rather than an instruction the agent has to follow voluntarily.
4. **Bounded re-review → this already exists natively as the Pre-Commit Remediation Loop.** This is Antigravity's strongest advantage over the other two: the loop is anchored to a deterministic gate (the pre-commit hook's exit code), not an open-ended "review again" instruction. Hook fails → agent refactors → tests re-run → retry commit → hook passes → done. It terminates on its own because the hook's pass/fail is unambiguous, unlike bounding a subjective LLM review by an iteration count.
5. **Security gate → `.agents/skills/stride-threat-model/SKILL.md`.** This project already has this skill (built earlier this session). On Antigravity it would be genuinely dynamically triggered — no translation needed, it's already in the native format.
6. **Human checkpoint → plan approval + permission mode, not just an agent promise.** Antigravity requires reviewing/approving the implementation plan before code generation (Day 1), and `strict`/`request-review` permission mode can require the same before a commit — so the checkpoint is enforced by configuration, not solely by the agent choosing to ask.
7. **Packaging → `.agents/skills/review-fix-loop/SKILL.md`.** Because Antigravity natively discovers and dynamically triggers everything under `.agents/skills/`, building this loop there gets genuine on-demand loading for free — the same file this session would have to re-explain each time becomes a real self-triggering skill in Antigravity.

**Net difference from the Claude Code mapping:** Antigravity turns steps 1, 3, 4, and 6 from "instructions the agent follows" into "behavior enforced by CONTEXT.md rules, permission modes, and hook exit codes" — structurally harder to skip. The tradeoff is step 2 (the review pass itself) is less turnkey — there's no equivalent to a single `/code-review` command with graduated effort levels; that discipline has to be designed in manually via Semgrep rules and prompt framing.

---

## 5. Portability: GitHub Copilot vs. This Setup

**Not portable** (specific to this agent/harness):
- The exact review skill with named effort levels (`medium`/`high`/`ultra`) and structured findings output.
- The Agent Skills system (`SKILL.md` with a `name`/`description` that self-triggers on relevance) — Copilot has no equivalent dynamic-loading primitive.

**Portable** (the underlying pattern, approximable in Copilot):
- **`.github/copilot-instructions.md`** — Copilot's equivalent of static context / `AGENTS.md`.
- **`.github/prompts/*.prompt.md`** (prompt files) — closest analog to a reusable "skill," though invoked explicitly rather than self-triggering on relevance.
- **Copilot coding agent + GitHub's built-in PR code review** — closest match to the review→fix→re-review loop: assign an issue, Copilot opens a PR, GitHub's automated review comments on it, and the coding agent can push follow-up commits addressing those comments — an iterate-until-clean cycle driven through PR review threads instead of a single session.
- **MCP support in VS Code's Copilot Chat (agent mode)** — tool wiring transfers directly since it's the same protocol.

The bounded-iteration/cost-control discipline (medium-by-default, cap at 3 passes, security gate once at the end, human checkpoint before merge) is **not** a built-in feature of either tool — it has to be enforced by whoever designs the workflow, regardless of which agent executes it. That's the actual reusable takeaway, independent of vendor.

> Note: Copilot's feature set moves quickly; verify against GitHub's current docs before committing to a Copilot-specific implementation of this workflow.

---

## Related
- [.agents/memory-bank/vibeEngineeringAssignments.md](../.agents/memory-bank/vibeEngineeringAssignments.md) — Day 4 entries 7 & 8, source of the Pre-Commit Remediation Loop / TDD Planning Gate / eval-loop patterns this workflow is built from.
- [.agents/skills/vibe-engineering-advisor/SKILL.md](../.agents/skills/vibe-engineering-advisor/SKILL.md) — pointer skill for applying these principles to harness/security design decisions generally.
