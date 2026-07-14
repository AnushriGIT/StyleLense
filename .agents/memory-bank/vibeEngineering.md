# Vibe Engineering — Knowledge Base

> **Source**: Synthesized from 5 whitepapers (May 2026, Google Cloud Next curriculum)
> **Purpose**: Portable procedural memory for the StyleLens project and any AI-assisted development workflow.

---

## 1. The New SDLC — From Syntax to Intent (Day 1)

### Core Paradigm Shift
The developer role has evolved from **writing code** to **managing the harness** — the scaffolding of state, tools, feedback loops, and enforceable constraints that wraps a raw AI model into a functioning agent. Code is now **disposable**; the specification is the durable asset.

### The Factory Model
Think of AI-assisted development as a **factory floor**:
- **Raw materials** = context (specs, code, docs, user intent)
- **Machine** = the LLM
- **Factory manager** = the developer, who controls the harness, not the weights
- **Output** = working software

The developer's job is Quality Control — writing specs, evaluating output, and tuning the harness.

### Context Engineering
The single most impactful lever for agent quality. Two types:

| Type | What it is | Examples |
|------|-----------|----------|
| **Static context** | Always loaded, every turn | `AGENTS.md`, system prompt, project conventions |
| **Dynamic context** | Loaded on-demand when needed | Agent Skills, RAG retrievals, MCP tool outputs |

**Key insight**: Active context is a **budget**, not a vessel. A 1M-token window can degrade at 50K tokens. Every token competes for attention.

### Memory Primitives for Agents
| Memory Type | Human Analog | Agent Implementation |
|-------------|-------------|---------------------|
| Episodic | "What happened" | Conversation history, session logs |
| Semantic | "What I know" | RAG, knowledge bases, embeddings |
| Procedural | "How to do it" | **Agent Skills** (the missing primitive until 2025) |

### Debugging Hierarchy
When an agent fails, investigate in this order:
1. **Context** — Is the right information reaching the model? (most common root cause)
2. **Tools** — Are tools configured correctly? Use MCP Inspector to debug.
3. **Prompts** — Only tweak prompts after ruling out #1 and #2.
4. **Model** — Swap models as a last resort.

---

## 2. Agent Tools & Interoperability — Protocols (Day 2)

### Protocol Map

| Protocol | Purpose | Analogy |
|----------|---------|---------|
| **MCP** (Model Context Protocol) | Agent ↔ Tool communication | "USB-C for AI tools" |
| **A2A** (Agent-to-Agent) | Agent ↔ Agent orchestration | Delegation, multi-agent coordination |
| **A2UI** (Agent-to-UI) | Agent → Dynamic UI generation | Declarative UI composition at runtime |
| **AP2** (Agent Payments Protocol) | Secure agent-driven payments | "The wallet with strict rules" |
| **UCP** (Universal Commerce Protocol) | Agent ↔ Merchant catalog/orders | "The brain that shops" |

### MCP Architecture
```
Host Process (IDE / Agent Runtime)
├── MCP Client (inside the host)
│   ├── MCP Server A  (local: filesystem, git)
│   ├── MCP Server B  (remote: database, API)
│   └── MCP Server C  (remote: cloud service)
```

**Key concepts**:
- **Resources**: Data the server exposes (read-only context)
- **Tools**: Functions the server exposes (actions with side effects)
- **Prompts**: Reusable prompt templates the server provides
- **Sampling**: Server asks the LLM to generate text (reverse direction)

**Security**: MCP servers are untrusted by default. Always validate tool inputs/outputs. Use the **MCP Inspector** for debugging.

### A2A (Agent-to-Agent)
- Uses **Agent Cards** (JSON metadata at `/.well-known/agent.json`) for discovery
- Supports **push notifications** and **streaming** for long-running tasks
- Introduces **Parts** (text, file, data, A2UI) as the communication unit

### A2UI — Generative UI
Two patterns:
1. **Intent-driven** (default): LLM generates the UI JSON from user intent
2. **Input-driven**: Deterministic template filled with data

**Security model**: Agents can only request components the renderer's **catalog** already trusts. No arbitrary code injection.

### Commerce Protocols
- **UCP**: Standardizes how agents discover products, build orders, check prices
- **AP2**: Handles secure payments with **mandates** (spending limits), cryptographic signing, and fraud prevention

---

## 3. Agent Skills — Procedural Memory (Day 3)

### What is a Skill?
A folder containing a `SKILL.md` file + optional `scripts/`, `references/`, `assets/` directories. Skills are the **first credible procedural memory primitive** for LLM agents.

### Progressive Disclosure (Critical Architecture)
Skills load in three tiers to minimize token cost:

| Level | What Loads | When | Token Cost |
|-------|-----------|------|-----------|
| L1: Metadata | `name` + `description` | Always (every turn) | ~50 tokens/skill |
| L2: Body | Full `SKILL.md` instructions | When description matches user intent | ~2,000 tokens |
| L3: Resources | `references/`, `scripts/`, `assets/` | When body references them | Variable |

**Math**: 50 skills × 50 tokens = ~2,500 tokens always loaded. vs. dumping everything = 150,000+ tokens of context rot.

### Skill Anatomy Template
```yaml
---
name: skill-name
description: |
  [What it does in one verb-led sentence.] Use this skill when the user
  [trigger phrase 1], [trigger phrase 2], or [trigger phrase 3].
  Do NOT use for [anti-trigger 1] or [anti-trigger 2].
---
# Skill Name
## When to use
- [Concrete scenario]
## When NOT to use
- [Out-of-scope scenario]
## Workflow
1. [Step]
2. [Step]
```

### The Five Rules
1. **One skill, one job** — If you can't describe it in one sentence, it's two skills.
2. **Descriptions are an interface** — The agent picks skills by reading descriptions. Vague = unused.
3. **Skills are dependencies** — Version, pin, review in PRs. No test = no capability.
4. **The right team owns the right skill** — Domain experts own domain skills.
5. **The agent runtime is interchangeable** — Don't tie skills to one runtime. Portability is the value.

### Skill vs. MCP vs. AGENTS.md
| Primitive | Purpose | Loading |
|-----------|---------|---------|
| `AGENTS.md` | Project conventions, stack, build commands | Always loaded (static) |
| Skill | Procedural know-how for a specific workflow | On-demand (dynamic) |
| MCP | Connection to external systems/data | On-demand (tools) |

**They compose**: A Skill teaches the agent *how to think* about work. MCP gives the agent *hands* to reach external systems.

### Evaluation Framework (4 Failure Modes)
1. **Trigger Failure** — Wrong skill fires, or correct one doesn't fire
2. **Execution Failure** — Skill triggers correctly but produces wrong output
3. **Token Budget Failure** — Large skill body causes context rot when co-loaded
4. **Regression** — New skill breaks previously working routing

**Graduation Tiers**:
- **Read-Only**: LLM-as-Judge eval; 90% trigger accuracy
- **Draft-Only**: Golden dataset of 20+ cases; human approval
- **Action-Allowed**: Full adversarial red-teaming; sustained pass^k

### Meta-Skills
Skills that write, evaluate, or improve other skills:
1. **Authoring** — Generate SKILL.md drafts from descriptions
2. **Trace harvesting** — Watch successful agent runs → propose skills
3. **Improvement** — Take failing eval cases → propose edits
4. **Library evolution** — Agent proposes new skills from production traffic

**Warning**: Meta-skills only work if evaluation is solid. Always start with manual authoring first.

### Skill Composition
- **DAG Orchestration** — Graph-based parallel execution with file-bus state passing
- **Capability Profiles** — Swappable tool/skill bundles for different execution states
- **Context Debt** — When skill descriptions accumulate "ALWAYS DO X" noise, models learn to ignore them. Instead: **Shift Intelligence Left** — push logic into scripts, not prompts.

---

## 4. Security & Evaluation — Trust as a Continuous Metric (Day 4)

### The 7-Pillar Security Architecture

| Pillar | Owner | Key Control |
|--------|-------|-------------|
| 1. Infrastructure | Cloud Engineers | Ephemeral sandboxes (gVisor), network egress governance |
| 2. Data | Data Architects | CMEK encryption, cross-tenant vector poisoning prevention |
| 3. Model | AI Engineers | System instructions as attested artifacts, prompt injection defense |
| 4. Application | Agent Developers | LLM firewalls, deterministic hooks, Centralized Agent Gateway |
| 5. Identity (IAM) | Identity Admins | SPIFFE IDs, ABAC, JIT token downscoping, Zero Ambient Authority |
| 6. Observability | SecOps | OpenTelemetry, Agent Behavioral Analytics, Runtime AgBOM |
| 7. Governance | Compliance | EU AI Act compliance, Algorithmic Impact Assessments, Logic Reviews |

### Key Security Concepts

**Effective Trust** — A continuous metric (not a one-time gate) evaluated across supply chain, identity, runtime behavior, and contextual associations.

**Confused Deputy** — An over-privileged agent tricked into executing unauthorized commands. Solution: dedicated agentic identity, not delegated user credentials.

**Slopsquatting** — Attackers publish malware under hallucinated package names. Solution: vetted registries + SBOM verification.

**Vibe Diff** — Before critical actions, translate generated code back to plain English so humans understand what they're approving.

### Agentic SecOps Triad
| Team | Role | Method |
|------|------|--------|
| **Red** | Attack simulation | Inject adversarial vibes, test prompt injection resilience |
| **Blue** | Monitoring | Agent Behavioral Analytics (ABA), Runtime AgBOM monitoring |
| **Green** | Auto-remediation | Stateful quarantine, auto-refactoring of vulnerabilities |

### Evaluation Framework (7 Dimensions)

**User-facing**:
1. Intent satisfaction — Did it build what the user *meant*?
2. Functional correctness — Does it build, run, pass tests?
3. Visual/behavioral correctness — Does the rendered output look right?
4. Cost & efficiency — Token spend, latency, correction count

**Internal**:
5. Code quality & convention matching
6. Trajectory quality — Did the agent take a sensible path?
7. Self-repair behavior — Does it recover from failures?

### Evaluation Methods
| Method | Best For |
|--------|----------|
| Automated testing (pytest, jest) | Functional correctness, code quality |
| LLM-as-Judge | Intent satisfaction, style, trajectory |
| Browser-based testing (Playwright) | Visual/behavioral correctness |
| Trajectory inspection (OpenTelemetry) | Trajectory quality, self-repair |
| Human review | Intent satisfaction, calibration |
| Online evaluation (sampled traffic) | All dimensions at sample rate |

### Practical Tips
- **Session prefix = intent rubric**: Derive acceptance criteria from the first 1-2 user messages
- **Judge the rendered artifact, not the code**: Use multimodal models on screenshots
- **Session convergence > turn accuracy**: Track turns-to-converge and abandonment rate
- **Mine user corrections**: Cluster "no, not like that" messages to find systematic agent gaps

---

## 5. Spec-Driven Development — Production-Grade Vibe Coding (Day 5)

### Core Principle
**"Vibe coding" is not "vibe in production."** Code is disposable; the specification is the durable asset. If you have a rock-solid spec, the entire codebase can be regenerated.

### Specification Format
Research shows (Ouyang et al., SkCC 2026):
- **Markdown + Conditional YAML** is optimal for Gemini
- YAML achieves 51.9% parsing accuracy for nested configs vs. 43.1% JSON
- Use clean Markdown headers for narrative, switch to YAML for schemas with nesting > 3

### BDD (Behavior-Driven Development)
Use **Gherkin** syntax: `Scenario / Given / When / Then`
- Forces State → Action → Outcome thinking
- Eliminates "vibe coding" guesswork
- Include: requirements, DB schemas, API specs, visual aids, edge cases

### Where Instructions Live
| Layer | Location | Purpose |
|-------|----------|---------|
| Chat | IDE panel | Ephemeral orchestration commands |
| Specs folder | `./specs/*.md` | Technical designs, BDD scenarios, API contracts |
| Skills | `.agents/skills/*/SKILL.md` | Reusable trigger-based workflows |
| Global profile | `~/.gemini/GEMINI.md` | Universal persona and style |
| Project spec | `./.gemini/GEMINI.md` | Project-specific DNA |
| Shared config | `.agents/AGENTS.md` | Cross-tool conventions |

### Execution Modes
1. **Project Generation (Architect)** — No YOLO mode; propose structure first; include version numbers
2. **Feature Generation (Builder)** — Match existing style; confirm multi-file changes via diffs
3. **Bug Fixing (Forensic)** — Evidence Prompting > Symptom Prompting; reproduce first
4. **Documentation Writing (Author)** — Docs are source of truth; use Google Style Docstrings
5. **Data Engineering (Librarian)** — Always show the SQL/command used

### Team Culture Evolution
- **Bundled Summaries**: Every PR includes AI-generated risk assessment
- **Conditional LGTM**: Auto-merge on green tests (cross-timezone)
- **Approval Fatigue**: AI users are 45% more likely to burn out; implement Digital Quiet Hours
- **Agent Insight Sessions**: Weekly team meetings to share agent-discovered patterns
- **No-Blame Culture**: Attribute bugs to broken processes, not individuals

### Zero-Trust Development
- **Sandboxing**: Execute all agent-generated code in ephemeral, network-isolated containers
- **Human-in-the-Loop**: Checkpoint gates for high-risk actions (deploy, DB schema changes)
- **Context Hygiene**: PII masking via `[[PLACEHOLDER]]` syntax + `resolve_context()` middleware
- **Policy Server**: Two-layer gating — Structural (YAML rules) + Semantic (LLM-based intent check)

### Code Review Tiers
| Tier | Name | Control Level | When to Use |
|------|------|--------------|-------------|
| 1 | Managed | Vendor defaults | Generic reviews, small teams |
| 2 | Hybrid | Your skill + CI Action | Team-specific criteria |
| 3 | Custom | Full ADK agent | Cross-PR context, compliance, graph-native understanding |

---

## Quick Reference: How This Applies to StyleLens

### Current Architecture Mapping
- **Embedding models** → MCP tool candidates (expose similarity search as a tool)
- **Fine-tuned DINOv2** → Best visual retrieval backbone (Recall@1: 16.80%)
- **CLIP** → Text-driven refinement layer (text↔image alignment)
- **Agent Skills** → `document-to-markdown`, `test-runner` already exist
- **Memory Bank** → This file + `systemPatterns.md` + `projectDocs.md`

### Recommended Next Steps
1. Create a **hybrid retrieval skill** that uses DINOv2 for visual search + CLIP for text refinement
2. Write BDD specs in `specs/` for the visual search API before generating code
3. Build a **code-review skill** for PRs using the Day 5 pattern
4. Add **eval coverage** for existing skills (trigger + execution + regression + token budget)
5. Expose the embedding pipeline as an **MCP server** for cross-tool reuse

---

## Endnotes & Source References

| Day | Title | Pages |
|-----|-------|-------|
| 1 | The New SDLC With Vibe Coding | 42 |
| 2 | Agent Tools & Interoperability | 49 |
| 3 | Agent Skills | 62 |
| 4 | Vibe Coding Agent Security and Evaluation | 41 |
| 5 | Spec-Driven Production Grade Development | 38 |

All source documents are located at:
`/Reference/Vibe Engineering/` in the StyleLens workspace.
