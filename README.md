# DiscoverSkill: Open-Ended Skill Discovery for LLM Agents

DiscoverSkill is a framework that automatically discovers and evolves a pool of reusable cognitive skills for LLM-based agents. Skills are markdown documents that the agent can invoke on demand; the framework continuously evaluates, refines, and invents new skills based on episode traces — no human skill engineering required.

Applied to **BabaIsYou**, a rule-manipulation puzzle game, DiscoverSkill raises agent solve accuracy from **21.4% → 32.8%** (+11.4 absolute, +53% relative) on a held-out set of 43 puzzles.

---

## How It Works

At each iteration the framework:

1. **Samples** a group of skills from the pool using a UCB exploration bonus to balance exploitation of high-value skills with exploration of under-tested ones.
2. **Runs** parallel agent episodes with the selected skills available as invokable reasoning modules.
3. **Assigns credit** to each skill via usage frequency × LLM-ranked quality — a skill must be both frequently invoked and demonstrably helpful to earn credit.
4. **Updates** each skill's long-term value estimate (V) via exponential moving average, only when the skill received non-zero credit (no causal signal → no update).
5. **Invents** new skills every K iterations by reflecting on failure traces, proposing targeted skills that address specific gaps in the current pool.

Skills that are domain-agnostic (e.g. *analogical reasoning*, *self-critique*) are naturally filtered out as their value decays toward zero. Domain-specific, actionable skills (e.g. *complete IS WIN with a subject noun-text*) rise to the top.

For full method details, see [`method_description.md`](method_description.md).  
For skill value analysis and results breakdown, see [`results_analysis_skill_values.md`](results_analysis_skill_values.md).

---

## Results

Evaluated on **43 BabaIsYou puzzles** (held-out test set):

| Condition | Solve Rate |
|---|---|
| No skill (baseline) | 21.4 ± 0.0% |
| Best discovered skill group | **32.8 ± 1.8%** |

The +11.4% gain comes entirely from skills invented and selected by the framework — no human-authored heuristics.

---

## Installation

```bash
git clone https://github.com/zksha/DiscoverSkill.git
cd DiscoverSkill
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, an OpenAI-compatible API key.

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# Use a single key for both agent and inventor
ANTHROPIC_API_KEY=sk-...

# Or set agent and inventor separately (e.g. different providers/models)
AGENT_API_KEY=sk-...
AGENT_BASE_URL=https://api.openai.com/v1

INVENT_API_KEY=sk-...
INVENT_BASE_URL=https://api.openai.com/v1
```

---

## Running

### Training (skill evolution)

```bash
./train.sh
```

Training hyperparameters used in our experiments:

| Parameter | Value | Description |
|---|---|---|
| `--iterations` | 10 | Number of evolution iterations |
| `--group-size` | 4 | Skills sampled per iteration |
| `--n-eval` | 10 | Episodes per iteration |
| `--n-train` | 10 | Training task count |
| `--max-steps` | 30 | Game steps per episode |
| `--invent-every` | 1 | Invent new skills every N iterations |
| `--n-new` | 3 | New skills proposed per invention round |
| `--alpha` | 0.1 | EMA learning rate for value update |
| `--tau` | 2.0 | Initial sampling temperature |
| `--agent-model` | `gpt-5-mini` | Model for agent episodes |
| `--invent-model` | `gpt-5` | Model for skill invention and credit assignment |
| `--seed` | 42 | Random seed |

Logs are written to `logs/train_<timestamp>.log`.

### Evaluation

```bash
./eval.sh
```

Runs the agent with the best discovered skill group on the full test set and reports solve rate.

---

## Project Structure

```
discoverskill/
├── main.py                  # Entry point
├── train.sh                 # Training script
├── eval.sh                  # Evaluation script
├── skillevo/
│   ├── agent.py             # LLM agent with skill invocation
│   ├── evolution.py         # Main evolution loop
│   ├── inventor.py          # Skill invention from traces
│   ├── credit.py            # Semantic credit assignment
│   ├── sampler.py           # UCB + softmax skill sampling
│   ├── skill_pool.py        # Pool management and value updates
│   ├── skill.py             # Skill data structure
│   └── envs/                # Environment wrappers (BabaIsYou, ...)
├── skill_repo/              # Discovered skill library (markdown + registry.json)
├── evolution_logs/          # Per-iteration history (history.jsonl)
└── logs/                    # Training and eval logs
```

---

## Analysis

The discovered skill pool shows a clear value split after ~12 iterations:

- **High-V skills** are concrete, domain-grounded sub-goal decompositions with BabaIsYou-specific vocabulary and clear executable triggers (avg credit 0.5–0.8).
- **Low-V skills** are domain-agnostic metacognitive heuristics (*analogise*, *self-critique*, *recall prior knowledge*) that receive near-zero credit despite frequent early sampling.

See [`results_analysis_skill_values.md`](results_analysis_skill_values.md) for the full breakdown.
