# Open-ended Semantic Skill Evolution — Method Description

## Overview

The propose an open-ended framework for evolving a pool of reusable cognitive skills for LLM-based agents. At each iteration, a subset of skills is selected and provided to the agent as on-demand reasoning modules. The agent's performance is used to update each skill's estimated value, and a separate LLM periodically invents new skills by reflecting on failure patterns. Over time, the pool converges toward skills that are genuinely useful for the target environment.

---

## 1. Skill Representation

Each skill $s_i$ consists of:
- A **markdown document** containing a detailed strategy description (the skill body)
- A **one-sentence description** used as a menu index — the agent reads only this line when deciding whether to invoke a skill
- A scalar **value estimate** $V(s_i) \in \mathbb{R}$, initialised to 0
- A visit count $N(s_i) \in \mathbb{Z}_{\geq 0}$, counting how many times $s_i$ has been included in an evaluated group

The value $V(s_i)$ is the central learning signal: it is updated via exponential moving average (EMA) of credit-weighted rewards, and drives sampling priority in subsequent iterations.

---

## 2. Skill Group Sampling

At each iteration $t$, we sample a group $G \subset \mathcal{S}$ of $k$ skills without replacement. Rather than sampling uniformly or purely by value, we apply a UCB-style exploration bonus that ensures under-sampled skills remain competitive.

### 2.1 UCB Score

$$U(s_i) = V(s_i) + c \cdot \sqrt{\frac{\log T}{N(s_i) + 1}}$$

where $T = \sum_j N(s_j)$ is the total number of skill evaluations across the pool, and $c$ is an exploration coefficient. The $N+1$ pseudocount ensures a freshly invented skill (with $N=0$) receives a strictly larger bonus than any skill with $N \geq 1$, guaranteeing that new skills are evaluated before the pool can discard them. Using $\log T$ in the numerator means the bonus grows with overall experience — skills that have not been revisited become increasingly attractive relative to well-sampled ones.

### 2.2 Softmax Sampling

UCB scores are converted to a probability distribution via temperature-controlled softmax:

$$p(s_i) = \frac{\exp(U(s_i) / \tau)}{\sum_j \exp(U(s_j) / \tau)}$$

where $\tau > 0$ is the temperature. High $\tau$ flattens the distribution (exploration); low $\tau$ concentrates mass on high-$U$ skills (exploitation). $k$ skills are drawn without replacement from $p$ to form the group $G$.

### 2.3 Adaptive Temperature

To avoid manual temperature scheduling, we adapt $\tau$ based on the Shannon entropy of the sampling distribution:

$$H_t = -\sum_i p_i \log p_i, \qquad H_{\max} = \log |\mathcal{S}|$$

After each iteration, $\tau$ is adjusted by a fixed step $\delta$:

$$\tau_{t+1} = \begin{cases} \min(\tau_t + \delta,\ \tau_{\max}) & \text{if } H_t < 0.4 \cdot H_{\max} \quad \text{(too concentrated)} \\ \max(\tau_t - \delta,\ \tau_{\min}) & \text{if } H_t > 0.85 \cdot H_{\max} \quad \text{(too uniform)} \\ \tau_t & \text{otherwise} \end{cases}$$

The target band $[0.4, 0.85] \cdot H_{\max}$ keeps the distribution meaningfully non-uniform (so high-V skills are favoured) while preventing collapse onto a single skill. This removes the need for a learning rate schedule for temperature.

---

## 3. Agent Episode

The $k$ selected skills are staged into an evaluation workspace as markdown files. The agent receives a system prompt containing:
1. The task description and available actions
2. A **skill menu**: one line per skill showing `[skill_id]  <description>`

On each step, the agent may optionally invoke a skill by writing `INVOKE_SKILL: <skill_id>`. The full skill body is then injected into the conversation as a user message, and the agent produces its action decision with the skill content in context. This two-phase invocation (description → decision to invoke → full content → action) means the agent pays a latency cost only for skills it deems relevant, analogous to a tool-use or retrieval pattern.

Each episode runs for up to $T_{\max}$ steps and returns a scalar reward $R \in [0, 1]$. We run $n_{\text{eval}}$ episodes in parallel and compute the mean reward:

$$\bar{R}_G = \frac{1}{n_{\text{eval}}} \sum_{j=1}^{n_{\text{eval}}} R^{(j)}$$

The agent also maintains a **step history** of the last 15 (observation, action) pairs, giving it access to the full transition sequence rather than just the most recent state. This allows the agent to detect repeated failures, measure the effect of previous actions, and make temporally informed decisions.

---

## 4. Semantic Credit Assignment

A core challenge in skill evolution is the **credit assignment problem**: given that $k$ skills were present and the group received reward $\bar{R}_G$, how much of that reward should be attributed to each individual skill?

We solve this with a two-stage process that combines objective usage evidence with LLM-based quality judgement.

### 4.1 Usage Score (objective gate)

We count how many times each skill was explicitly invoked across all $n_{\text{eval}}$ episodes, then normalise by the maximum count:

$$\text{usage}_i = \frac{\text{count}(s_i)}{\max_j \text{count}(s_j)}$$

A skill that was never invoked receives $\text{usage}_i = 0$. This acts as a hard gate: a skill that the agent chose not to use cannot receive credit, regardless of how the episode went.

### 4.2 Rank Score (LLM quality judgement)

Among the invoked skills, we ask a separate LLM to rank them by contribution quality, given the full episode traces (observation → skill invoked → reasoning → action for each step) and the final reward. The ranker sees concrete evidence of each skill's effect on the agent's reasoning and can distinguish between a skill that was mechanically invoked versus one that visibly improved the decision.

The ranking is converted to a continuous score:

$$\text{rank\_score}_i = \frac{k - \text{rank}_i}{k}$$

where $\text{rank}_i \in \{1, \ldots, k\}$ and rank 1 is the most valuable. This gives a linear interpolation from $1.0$ (best) to $1/k$ (worst among invoked skills), with all non-invoked skills at $0$.

### 4.3 Combined Credit

$$C(s_i) = \text{usage}_i \times \text{rank\_score}_i \in [0, 1]$$

The multiplicative combination enforces that **a skill must be both used frequently and used effectively** to receive high credit. A skill invoked many times but ranked last still gets low credit; a skill ranked first but invoked only once also gets modest credit. This prevents gaming: an agent cannot inflate a skill's credit by invoking it repeatedly without the invocations actually helping.

---

## 5. Value Update

Each skill's value is updated via EMA of its credit-weighted reward:

$$r_i = C(s_i) \cdot \bar{R}_G$$

$$V(s_i) \leftarrow \begin{cases} (1 - \alpha) \cdot V(s_i) + \alpha \cdot r_i & \text{if } C(s_i) > 0 \\ V(s_i) & \text{if } C(s_i) = 0 \end{cases}$$

where $\alpha \in (0, 1)$ is the EMA learning rate.

**Why skip the update when $C = 0$?** A skill can be absent from the group ($C = 0$ trivially) or present but never invoked by the agent. In either case, there is no causal evidence linking the skill to the episode outcome — the reward could be entirely due to the other $k-1$ skills. Applying an update from a zero-credit signal would introduce noise: if the group succeeded, the non-contributing skill would be incorrectly rewarded; if it failed, it would be incorrectly penalised. Skipping the update preserves $V$ until genuine evidence arrives.

**Convergence behaviour.** Over many iterations, $V(s_i)$ converges to $\mathbb{E}[C(s_i) \cdot R_G \mid C(s_i) > 0]$ — the expected credit-weighted reward conditional on the skill actually contributing. This is a natural utility estimate: it is high only for skills that are invoked often, invoked in episodes where the group succeeds, and rated highly by the credit ranker.

The visit count $N(s_i)$ is incremented whenever $s_i$ is included in a group, regardless of whether it was invoked. This ensures the UCB exploration bonus correctly reflects how many opportunities the skill has had, not just how often it was used.

---

## 6. Skill Invention

Every $K$ iterations, an LLM inventor reflects on recent experience and proposes $n_{\text{new}}$ new skills. The inventor receives:

- **Environment description**: mechanics and task categories of the current domain
- **Last evaluated group**: each skill's description, post-update $V$, and credit $C$
- **All episode outcomes**: task names and rewards for all $n_{\text{eval}}$ episodes
- **Sampled episode traces**: up to 3 complete (observation, skill invoked, action, reasoning) traces, randomly selected to provide diverse examples
- **Full pool summary**: all existing skill names and descriptions, to avoid duplicates

The inventor is prompted to identify failure patterns — which tasks were consistently unsolved, where the agent's reasoning went wrong, which skills were helpful and which were ignored — and to propose skills that address specific gaps. Crucially, skill descriptions must be **≤15 words starting with an action verb**, because the agent reads only the description when deciding whether to invoke a skill. Vague or generic descriptions produce skills that are never invoked and consequently never receive credit or V updates.

New skills are added to the pool with $V = 0$ and $N = 0$. The UCB bonus is therefore maximised for freshly invented skills ($T_{\text{current}} \gg 0$ while $N = 0$), guaranteeing they are sampled and evaluated before the pool can effectively ignore them.

**Design rationale for trace-grounded invention.** Early iterations produce generic skills (analogical reasoning, self-critique) because the agent rarely solves tasks and the traces carry little signal about what specific capability is missing. As the pool accumulates domain-specific skills and the agent begins solving more tasks, the traces become richer and the inventor can identify more precise failure modes. This creates a positive feedback loop: better skills → better traces → better invention → better skills.

---

## 7. Full Iteration Loop

```
for t = 1, 2, ..., T:

    1. SAMPLE GROUP
       U(s_i) = V(s_i) + c · sqrt(log(T_total) / (N(s_i) + 1))
       p(s_i) = softmax(U / τ)
       G = sample k skills without replacement from p

    2. STAGE EVAL WORKSPACE
       Write skill .md files for G to eval directory

    3. RUN EPISODES (parallel)
       For each of n_eval task instances:
           Run agent with G in context (up to T_max steps)
           Record per-step (obs, action, skills_invoked, reasoning)
       Compute mean reward R̄_G

    4. CREDIT ASSIGNMENT
       For each episode, count INVOKE_SKILL calls per skill
       Aggregate usage counts across all episodes
       usage_i = count_i / max_count
       LLM ranks invoked skills given traces and R̄_G
       rank_score_i = (k - rank_i) / k
       C(s_i) = usage_i × rank_score_i

    5. VALUE UPDATE
       For s_i in G:
           r_i = C(s_i) · R̄_G
           N(s_i) += 1
           if C(s_i) > 0:
               V(s_i) ← (1-α)·V(s_i) + α·r_i

    6. ADAPTIVE TEMPERATURE
       H = -Σ p_i log p_i
       Adjust τ based on H vs. target band [0.4, 0.85]·log|S|

    7. SKILL INVENTION  (if t mod K == 0)
       Inventor LLM proposes n_new new skills from traces + pool summary
       Add to pool with V=0, N=0

    8. PERSIST
       Save registry.json (V, N, logs per skill)
       Append to history.jsonl
```

---

## 8. Design Principles

| Component | Design choice | Rationale |
|---|---|---|
| UCB bonus | $\sqrt{\log T / (N+1)}$ | Classic UCB1 structure; guarantees new skills are explored before the pool can exclude them |
| Softmax sampling | Temperature-controlled, not argmax | Enables stochastic exploration; group diversity is important since skills interact |
| Adaptive $\tau$ | Entropy-based feedback | Avoids manual scheduling; automatically re-explores when the pool has converged prematurely |
| Credit gate ($C=0$ → no update) | Skip V update when skill not invoked | No causal signal → no update; prevents noise from non-contributing skills |
| Multiplicative credit | $C = \text{usage} \times \text{rank}$ | Requires both frequency and quality; prevents inflation from mechanical invocations |
| Skill description constraint | ≤15 words, starts with verb | Agent reads only the description to decide invocation; vague descriptions → zero invocations → zero credit |
| Trace-grounded invention | Full (obs, action, skill) trajectory sent to inventor | Inventor can identify exactly where reasoning broke down, not just that reward was low |
| N+1 pseudocount | $N(s_i) + 1$ in UCB denominator | Prevents division by zero; strictly orders unvisited > visited-once in exploration priority |
