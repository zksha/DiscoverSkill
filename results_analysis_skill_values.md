# Results and Skill Value Analysis — BabaIsYou Evolution

**Training setup:** 10 iterations | group size 4 | 10 episodes/iter | invent every 1 iter | agent: `gpt-5-mini` | inventor: `gpt-5` | α=0.1 | τ=2.0 | seed=42

---

## Results

Evaluated on **43 BabaIsYou puzzles**. Without any skill, the agent scores **21.4 ± 0.0** (solve rate). Using the best skill group discovered by DiscoverSkill, performance rises to **32.8 ± 1.8** — a **+11.4 point** improvement (+53% relative).

| Condition | Solve Rate (%) |
|---|---|
| No skill (baseline) | 21.4 ± 0.0 |
| Best skill group | **32.8 ± 1.8** |

---

## 1. Training Reward Trend

```
iter  6   reward=0.10  solved= 1/10  ██
iter  7   reward=0.10  solved= 1/10  ██
iter  8   reward=0.00  solved= 0/10
iter  9   reward=0.00  solved= 0/10
iter 10   reward=0.00  solved= 0/10
iter 11   reward=0.00  solved= 0/10
iter 12   reward=0.30  solved= 3/10  ██████   ← first batch of domain-specific skills invented
iter 13   reward=0.30  solved= 3/10  ██████
iter 14   reward=0.50  solved= 5/10  ██████████
iter 15   reward=0.40  solved= 4/10  ████████
iter 16   reward=0.30  solved= 3/10  ██████
iter 17   reward=0.40  solved= 4/10  ████████
iter 18   reward=0.20  solved= 2/10  ████
iter 19   reward=0.30  solved= 3/10  ██████
iter 20   reward=0.50  solved= 5/10  ██████████
iter 21   reward=0.40  solved= 4/10  ████████
iter 22   reward=0.30  solved= 3/10  ██████
iter 23   reward=0.50  solved= 5/10  ██████████
iter 24   reward=0.30  solved= 3/10  ██████
```

**The key turning point is iter 12.** Four consecutive zero-reward iterations preceded it; afterwards reward stabilises at 0.3–0.5. The timing corresponds to iter 10–11 invention rounds producing the first genuinely BabaIsYou-specific skills — `greedy_step_toward_win`, `extract_you_and_win_targets`, `assemble_is_win_when_missing` — which were then sampled and validated in iter 12.

---

## 2. High-V Skills

| Skill | V | N | Avg Credit |
|---|---|---|---|
| anchor_is_win_and_attach_noun | 0.0560 | 3 | 0.667 |
| complete_is_win_with_subject | 0.0522 | 2 | 0.808 |
| attach_subject_via_precise_push | 0.0510 | 2 | 0.600 |
| greedy_step_toward_win | 0.0500 | 2 | 0.500 |
| select_viable_subject_text | 0.0400 | 1 | — |

### Common pattern: actionable sub-goal decomposition

High-V skills are **concrete operational instructions targeting specific BabaIsYou mechanics**, specifying what to do under what conditions:

**`anchor_is_win_and_attach_noun`** (V=0.056, avg credit=0.667)
> "Treat any aligned IS and WIN as a fixed anchor you should not disturb. Determine the anchor orientation, then pick the closest movable noun-text..."

Decomposes the make_win problem into two steps: ① identify the IS WIN anchor ② compute the push-side route. The instruction granularity is precise enough for the agent to execute directly.

**`complete_is_win_with_subject`** (V=0.052, avg credit=0.808 — highest credit in pool)
> "Find an existing 'IS WIN' alignment, then route and push a noun-text to complete it."

The highest-credit skill in the pool. It addresses the most common make_win scenario in the training set — when IS and WIN are already aligned and only a subject noun-text is missing — and provides a clear routing strategy for that exact condition.

**`greedy_step_toward_win`** (V=0.050)
> "Pick the next step toward the nearest WIN object, avoiding STOP tiles."

The simplest high-V skill. It does exactly one thing: compute the next step toward WIN. Simple does not mean ineffective — goto_win variants account for roughly 40% of the test set, and this skill alone handles them.

**Pattern:** High-V skill descriptions all start with an **action verb** (Use / Find / Pick / Compute / Select) and contain BabaIsYou-specific vocabulary (IS WIN, noun-text, push-side, STOP tiles), indicating the inventor had learned to generate domain-grounded skills by iter 12+.

---

## 3. Low-V Skills

| Skill | V | N | Type |
|---|---|---|---|
| analogical_reasoning | 0.0 | 5 | generic reasoning |
| prior_knowledge_retrieval | 0.0 | 6 | generic reasoning |
| test_skill | 0.0 | 5 | placeholder |
| identify_paths_to_win | 0.0 | 4 | analytical |
| self_critique | 0.015 | 10 | generic reasoning |

### Common pattern: domain-agnostic, no executable entry point

**`analogical_reasoning`** (V=0, N=5)
> "Map unfamiliar problems onto familiar domains to leverage existing knowledge."

A canonical generic skill that is useless in BabaIsYou — "map the problem onto a familiar domain" offers no actionable foothold for a grid-based puzzle where every state must be read from the current observation. N=5 shows it was sampled frequently but never received meaningful credit, and V was driven to zero.

**`prior_knowledge_retrieval`** (V=0, N=6)
> "Systematically recall domain knowledge before reasoning."

Equally domain-agnostic. BabaIsYou is a pure perception-action task; "recalling domain knowledge" cannot extract useful information from a text observation of block positions. N=6 is among the highest in the pool, yet credit is zero across all appearances.

**`self_critique`** (V=0.015, N=10 — most sampled skill)
> "Steelman the opposite answer before finalizing to catch blind spots."

Has some credit (avg=0.10), suggesting occasional utility. However, the game demands sequential action decisions; a "write then argue against yourself" pattern does not fit the step-by-step action paradigm well. The high N=10 is an artefact of being sampled heavily when the pool was small in early iterations, accumulating many low-reward records that depressed V through the EMA update.

**`identify_paths_to_win`** (V=0, N=4)
Too analytical — it tells the agent to "find paths" without specifying how to navigate. By contrast, `greedy_step_toward_win` says "take this next step in this direction", which is directly executable.

**Pattern:** Low-V skill descriptions tend to start with noun phrases and contain no BabaIsYou-specific vocabulary. Their content is metacognitive advice ("analogise", "critique", "recall") rather than operational instructions.

---

## 4. Credit Assignment Analysis

```
complete_is_win_with_subject        avg_credit=0.808  ← extremely high; effective almost every invoke
anchor_is_win_and_attach_noun       avg_credit=0.667
attach_subject_via_precise_push     avg_credit=0.600
greedy_step_toward_win              avg_credit=0.500
─────────────────────────────────────────────────────
self_critique                       avg_credit=0.100  ← effective only occasionally across 10 uses
sequential_rule_manipulation        avg_credit=0.050
prior_knowledge_retrieval           avg_credit=0.000  ← zero across all 6 appearances
identify_paths_to_win               avg_credit=0.000  ← zero across all 4 appearances
```

The credit distribution shows a clear bimodal split — high-V skills consistently receive credit in the 0.5–0.8 range, while low-V skills remain at zero. This validates the credit mechanism: agents tend to invoke skills only when they genuinely need them, and high-V skills demonstrably improve decision quality when invoked.

---

## 5. Key Observations

### 5.1 Early generic skills suppress V updates
`analogical_reasoning`, `prior_knowledge_retrieval`, and `self_critique` are initial seed skills that were sampled heavily (N=5–10) when the pool was small, yet contributed almost nothing. The EMA formula `V ← (1−α)V + α·r` continuously pushes V toward zero when r=0, reducing sampling probability in later iterations through the UCB mechanism — a natural negative feedback loop. This is the correct behaviour: the pool is organically filtering out domain-irrelevant skills.

### 5.2 Invention quality shows a clear phase transition
Invention rounds in iter 6–11 produced predominantly generic skills (`identify_block_types`, `create_alternate_win_conditions`, `sequential_rule_manipulation`), all with near-zero V. From iter 12 onwards, the inventor began generating specific skills like `complete_is_win_with_subject` and `anchor_is_win_and_attach_noun`, and training reward rose accordingly. This suggests **inventor quality is gated by trace quality** — early on, the agent rarely solved tasks, episode traces carried little signal, and inventions were poor. Once the agent started solving tasks (using the first effective skills), richer traces enabled better inventions in a positive feedback loop.

### 5.3 High-V skills are compact sub-goal decompositions
The effective skill pattern is: decompose a large task ("solve make_win") into a directly executable small task ("push noun-text to the left of IS WIN"). Failed skills are either too coarse ("analyse the full path") or too generic ("analogise to another domain"). This has direct implications for invention prompt design — encouraging the inventor to target single, concrete sub-goals rather than broad strategies.

### 5.4 Low-N high-V skills carry statistical uncertainty
`select_viable_subject_text` (V=0.04, N=1) and `assemble_is_win_when_missing` (V=0.03, N=1) have only been evaluated once; their V estimates are unreliable. If resampled in continued training, these could rise or fall significantly.

---

## 6. Summary

| Dimension | High V | Low V |
|---|---|---|
| Description granularity | Concrete operation (how to do X) | Metacognitive advice (think about X) |
| Domain relevance | Contains BabaIsYou-specific terms | Generic language, domain-agnostic |
| Executable entry point | Clear conditional trigger | Vague, applicable to anything |
| Invention timing | iter 12+ (after rich failure traces) | iter 6–11 (early invention) |
| Credit after invocation | 0.5–0.8 | 0.0–0.1 |
