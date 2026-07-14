# Vibe Engineering — Assignment Walkthroughs

Practical codelab walkthroughs distilled from the links in [AssignmentSamples.md](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/Reference/Vibe%20Engineering/AssignmentSamples.md). Each entry captures the concrete steps and expected outcome so the exercise can be replayed without re-reading the source codelab.

---

## Day 1

### 1. Getting Started with Google Antigravity (IDE Setup)
*Source: `codelabs.developers.google.com/getting-started-google-antigravity`*

- Install Antigravity for your OS, authenticate with a Google account, accept security policies, pick a theme, optionally install Developer Tools plugins.
- Create a project folder (e.g. `$HOME/agy2-projects/my-first-project`), start a new project via **Select Project → New Project**, add the folder, set default security settings, name and create it.
- Start a conversation, rename threads via the three-dot menu, open parallel conversations with **+**, tune permissions/tool access in **Project Settings**.
- Explore `/` slash commands (e.g. `/browser` for automated web tasks), set up recurring jobs under **Schedule**, wire MCP servers under **Settings → Customizations**.
- Request code generation from a detailed prompt, review the agent's implementation plan before it executes, use the **Auxiliary Pane** to inspect generated artifacts, and open the full Antigravity IDE for editing.
- Create a project skill under `.agents/skills/` with a `SKILL.md`, then validate it by triggering a matching request.

**Expected outcome:** A configured local agent dev environment with multiple conversation threads, generated artifacts/plans, and at least one scheduled task and one custom skill working.

### 2. Build a Web App in AI Studio and Deploy to Cloud Run
*Source: `codelabs.developers.google.com/deploy-from-aistudio-to-run`*

- Create a (personal, non-workspace) Google account, open Google AI Studio, review ToS, configure model/framework in Settings.
- Describe the desired app in detail, click **Build**, wait ~2–3 min, choose design options, test in-browser, refine with follow-up prompts.
- Click **Publish**, pick a Cloud Project (or the default Gemini project), fill in billing if prompted, click **Publish your app**, wait for the Cloud Run deploy, then open the App URL to confirm it's live.
- Cleanup: click **Unpublish app** to stop charges.

**Expected outcome:** A prototype web app generated from a natural-language prompt and deployed to a live Cloud Run URL with no manual infra work.

---

## Day 2

### 3. Getting Started with Antigravity CLI
*Source: `codelabs.developers.google.com/antigravity-cli-hands-on`*

- Create a project folder, install the CLI (curl on macOS/Linux, PowerShell on Windows), verify with `agy --version`.
- Launch `agy`, log in via Google OAuth, pick a color theme, accept ToS, trust the workspace.
- Send a first prompt, run `/help` to see commands/shortcuts, run `/config` to review tool-permission modes (`request-review`, `proceed-in-sandbox`, `always-proceed`, `strict`) — settings persist in `~/.gemini/antigravity-cli/settings.json`.
- Use `agy -p "prompt"` for one-shot non-interactive answers, `agy models` / `agy --model "name"` to pick a model, `!` to toggle shell mode.
- Build a small web app end-to-end (approving artifacts/file permissions as prompted), inspect plans/tasks with `/artifact`, add a `.gitignore` and push the result to GitHub.

**Expected outcome:** Comfort with interactive vs. non-interactive CLI modes, permission levels, and a working app pushed to GitHub.

### 4. Google Developer Knowledge MCP Server in Antigravity
*Source: `codelabs.developers.google.com/developer-knowledge-mcp-antigravity`*

- Create/select a Cloud project, enable the **Developer Knowledge API**, generate an API key restricted to that API.
- Edit `~/.gemini/config/mcp_config.json` to add the server (`https://developerknowledge.googleapis.com/mcp`) with the API key in headers; save.
- Confirm `google-developer-knowledge` shows up in Antigravity's MCP server list (refresh method varies by version).
- Ask the agent natural-language questions about Workspace/Cloud APIs/codegen; approve tool use on first call.

**Expected outcome:** Antigravity answers with live, accurate Google documentation via MCP instead of relying on model memory or manual API calls.

---

## Day 3

### 5. How Skills Work in Antigravity
*Source: `codelabs.developers.google.com/getting-started-with-antigravity-skills`*

Four progressively richer skill patterns, each in an isolated test dir:

- **Level 1 — Basic Router (`git-commit-formatter`):** init a git repo with a sample `auth.py`, edit it, ask the agent to commit using the skill → verifies Conventional Commits output like `feat(auth): implement google login`.
- **Level 2 — Asset Utilization (`license-header-adder`):** ask the agent to create a new `.py` file; skill auto-reads `HEADER_TEMPLATE.txt` and prepends a correctly-commented license header.
- **Level 3 — Few-Shot Learning (`json-to-pydantic`):** provide `product.json`; ask for a Pydantic model; skill uses example input/output pairs to produce PascalCase classes with proper types.
- **Level 4 — Procedural Logic (`database-schema-validator`):** provide a `bad_schema.sql` breaking three rules (DROP statements, camelCase tables, missing PKs); skill runs `python scripts/validate_schema.py` (deterministic, not LLM judgment) and reports exact violations + fixes.
- **Bonus — Agents CLI skills lifecycle:** `uvx google-agents-cli setup` → `agents-cli create weather-assistant --prototype --yes` → `agents-cli run "How are you?"` → `agents-cli playground` (local web UI).

**Expected outcome:** Understanding of the skill "ladder" — instruction-only → templated assets → few-shot examples → deterministic script delegation — and how each avoids relying on pure LLM judgment where precision matters.

### 6. Build Agents with Agents CLI and ADK 2.0
*Source: `codelabs.developers.google.com/agents-cli-adk-lifecycle`*

- Set up auth (Gemini API key or Cloud credentials).
- `uvx google-agents-cli setup` installs the CLI + 7 domain skills into `~/.agents/skills/`.
- Prompt Antigravity to scaffold a graph-workflow agent (ADK 2.0) → get `app/agent.py` + supporting files; ask the agent to explain the workflow's nodes/edges/`LlmAgent` components.
- Run `agents-cli lint` for import/syntax/format checks.
- Start the local dev server → playground at `http://127.0.0.1:8080/dev-ui/?app=app`; test routing with in-domain and out-of-domain queries; edit code and confirm auto-reload.
- Run a single-turn inference via `agents-cli run`; clean up server + files when done.

**Expected outcome:** A working local ADK 2.0 graph agent, validated via lint, playground, and CLI single-turn execution.

---

## Day 4

### 7. Ambient Expense-Approval Agent (HITL + Local Evals)
*Source: `codelabs.developers.google.com/vibecode-ambient-expense-agent`*

- Install Antigravity, run `uvx google-agents-cli setup`, verify with `agents-cli info`.
- Scaffold `ambient-expense-agent` from the ADK starter template; set up `.env` auth (AI Studio key or `gcloud auth application-default login`).
- Build the graph workflow: `auto_approve` node (pure Python, <$100) → `review_agent` node (Gemini analysis, ≥$100) → human-in-the-loop node using `RequestInput` for approve/reject; keep threshold/model choice in config.
- Add a `security_screen` node **before** the LLM: regex-based PII redaction (SSNs, card numbers) + prompt-injection detection that routes suspicious payloads straight to a human, bypassing the model.
- Generate a Makefile (`install`, `playground` targets); `make playground` to test interactively — submit a sample expense, watch it traverse the graph to the HITL pause, click Approve/Reject.
- Convert to event-driven: add a FastAPI entrypoint (`fast_api_app.py`) on port 8080 exposing `/apps/expense_agent/trigger/pubsub`, decoding Base64 Pub/Sub payloads, normalizing subscription paths to session IDs, `otel_to_cloud=False` with standard logging.
- Run `make playground` again and use provided `curl` commands to simulate Pub/Sub events — confirm low-value auto-approves and session tracking shows in the Dev UI (`http://localhost:8080/dev-ui/?app=expense_agent&userId=test-sub`).
- Security test: send a high-value expense containing an SSN + injection attempt → confirm PII is redacted, the LLM is bypassed, and the workflow pauses at human review with a security flag.
- Evaluate: create a 5-scenario synthetic dataset (`tests/eval/datasets/basic-dataset.json`), a trace generator (`tests/eval/generate_traces.py`) that automates HITL decisions, and `eval_config.yaml` with two LLM-as-judge metrics (routing correctness, security containment). Run `make generate-traces && make grade` → scorecards land in `artifacts/grade_results/`.
- Cleanup: stop services, delete `.env`/keys, optionally `uv tool uninstall google-agents-cli`.

**Expected outcome:** A stateful ADK 2.0 graph agent with code-driven routing, a pre-LLM security screen, an ambient Pub/Sub trigger, and automated LLM-as-judge evaluation — this is the same expense agent later deployed in Day 5.

### 8. Secure AI Code: Threat Scans, Guards, Testing
*Source: `codelabs.developers.google.com/secure-agentic-coding`*

- `~/secure-agent-lab` + git init + `uv` venv; install `agents-cli`; `export GEMINI_API_KEY=...`.
- `agents-cli scaffold` a shopping-assistant ADK 2.0 project with a discount-redemption tool (single-use code validation); add `pre-commit`, `pre-commit-hooks`, `semgrep` to `pyproject.toml`; **intentionally** hardcode a mock API key to prove detection works.
- `agents-cli lint`; review the generated `agent.py` (LlmAgent + Workflow).
- Write `.agents/CONTEXT.md` — "paved roads" doc covering tool input validation and shell-execution restrictions, plus a "Pre-Commit Remediation Loop" note.
- `.semgrep/rules.yaml` with a `AIzaSy[A-Za-z0-9_\-]*` regex rule to catch hardcoded Google API keys; generate `.pre-commit-config.yaml` running Semgrep with `--error`; `pre-commit install`.
- `.agents/hooks.json` with a `PreToolUse` matcher on `run_command` + `.agents/scripts/validate_tool_call.py` to block destructive commands (e.g. `rm -rf /`).
- `.agents/skills/stride-threat-model/SKILL.md` defining a STRIDE methodology; run it to produce `threat_model.md`.
- Add a planning-gate rule to `CONTEXT.md` requiring a security section in every implementation plan before code generation (TDD-style gate).
- `tests/test_agent.py`: outcome-based pytest suite for single-use redemption, invalid-code rejection, guest-account blocking; run `pytest` green.
- `git add . && git commit` → Semgrep blocks the commit on the hardcoded key → let Antigravity refactor the key out to an env var → confirm tests still pass → commit succeeds.
- `agents-cli playground` at `http://127.0.0.1:8080` to manually test discount redemption; cleanup with Ctrl+C and `rm -rf ~/secure-agent-lab`.

**Expected outcome:** A pre-commit gate (Semgrep) that catches hardcoded secrets before they reach the repo, a STRIDE threat model, hook-based blocking of destructive shell commands, and a security-first planning gate baked into the agent's workflow.

---

## Day 5

### 9. [Optional] Deploy the Expense Agent to Agent Runtime
*Source: `codelabs.developers.google.com/enterprise-cloud-scale-deploying-the-expense-agent-to-agent-runtime-on-google-cloud`*

- Set up the Cloud project; enable AI Platform, Cloud Trace, Cloud Build, Agent Registry APIs.
- `uvx google-agents-cli setup`; scaffold the Ambient Expense Agent prototype (same graph as Day 4: auto-approve <$100, human review ≥$100).
- Generate production wrapper `app/agent_runtime_app.py` + `deployment_metadata.json`.
- `uv lock` then `agents-cli deploy --dry-run` to catch dependency conflicts before a real deploy.
- Deploy for real (~5–10 min) → get a live endpoint URL.
- Smoke-test with a $50 payload (auto-approves) and a $150 payload (pauses for human review).
- Check Cloud Trace/Logging for execution telemetry.
- Confirm auto-registration via `agents-cli publish gemini-enterprise --list`.
- Optional cleanup: delete the deployed engine, metadata file, and container images.

**Expected outcome:** The Day 4 expense agent running as a hosted, traceable, organization-discoverable service on Agent Runtime instead of only locally.

### 10. [Optional] Front-End Dashboard Linked to the Cloud Agent
*Source: `codelabs.developers.google.com/vibecode-frontend-with-antigravity`*

- Confirm ADK skills are loaded, Cloud project connected, required APIs enabled (AI Platform, Cloud Run, Pub/Sub, Cloud Build), and the Day 5 #9 agent is live on Agent Runtime.
- Build a `submission_frontend/` FastAPI service with: `GET /` (glassmorphic HTML dashboard), `GET /api/pending` (queries ADK Session Service for unresolved approvals), `POST /api/action/{session_id}` (resumes a paused agent session). Config (project ID, Agent Runtime ID) comes from env vars.
- Deploy the dashboard to Cloud Run as `expense-manager-dashboard` with public unauthenticated access; grant the runtime service account the IAM permissions it needs.
- Create Pub/Sub topics `expense-reports` and a dead-letter topic `expense-reports-dead-letter`.
- Create a `pubsub-invoker` service account; wire a push subscription with OIDC auth (`--push-no-wrapper`), a 10-minute ack deadline, and 5-attempt failure routing to the DLQ.
- End-to-end test: publish a low-value expense (auto-approved, doesn't surface on the dashboard) and a high-value one (appears on the dashboard, actionable); approve it from the UI and check the trace in Agent Runtime.
- Cleanup: delete the Cloud Run service, Pub/Sub topics/subscriptions, and service accounts.

**Expected outcome:** An event-driven loop — Pub/Sub → deployed agent → manager dashboard — where only expenses needing human judgment surface in a web UI, closing the loop opened in Day 4/5 #9.
