**Filename:** `docs/60_agent_handbook.md`

---

# 60 – Handbook Design, Goal-State JSON & Cycle Scheduler

The *handbook* is the contract between the human operator, Gemma, and the deterministic executor.
It tells Gemma **what job to do**, **what tools are legal**, and **what “done” looks like**—nothing more.
You can create a new handbook in plain Markdown (or YAML front-matter) to launch a fresh autonomous run without touching code.

---

## 1  Handbook File Anatomy

```markdown
# -- handbook: gap_kd_epilepsy.md ---------------------------------

Primary_goal: seek_gap                       # mandatory
Seed_query:   "ketogenic diet AND epilepsy"  # optional
Tools_allowed:
  - search_pubmed
  - run_pico
  - find_existing_sr
  - propose_alternative_pico
  - prisma_check
Exploration: true           # Gemma may use ripple mode
Success:
  eligible_trials  >= 6
  AND PRISMA_compliant == false
Stop_after: 1              # #success clusters before stop
-------------------------------------------------------------------
```

### 1.1  Field semantics

| Key             | Meaning                                                             |
| --------------- | ------------------------------------------------------------------- |
| `Primary_goal`  | `"seek_gap"`, `"conduct_SR"`, `"manual_task"` …                     |
| `Seed_query`    | Initial PubMed semantic search (passed to `search_pubmed`).         |
| `Tools_allowed` | Whitelist of heavy tools from `docs/70_tools.md`.                   |
| `Exploration`   | `true` allows Gemma to call *ripple* expansion; else semantic only. |
| `Success`       | Boolean expression using metrics available in cluster manifest.     |
| `Stop_after`    | Agent halts after X clusters meet `Success`.                        |

No other keys are recognised; unknown keys cause executor error at load time.

---

## 2  Goal-State JSON

`goal_state.json` is updated and persisted **each cycle**.

```json
{
  "goal_id": "6a4f46a1b2e6",
  "primary_goal": "seek_gap",
  "topic": "ketogenic diet epilepsy adults",
  "status": "in_progress",
  "handbook_path": "handbooks/gap_kd_epilepsy.md",
  "cluster_id": null,
  "subgoals": [],
  "history": [
     {"cycle":0,"event":"created"},
     {"cycle":5,"event":"pivoted","new_goal":"conduct_SR"}
  ]
}
```

* `goal_id` = `sha1(handbook_path + start_ts)[:12]`.
* `status` transitions: `in_progress` → `stopped`.
* Append-only `history` records deformable events (pivot, stop, error).

Gemma is **allowed** to mutate every top-level field except `goal_id` and `history`.
Executor appends history automatically when it detects changes.

---

## 3  Tool Invocation Grammar

Gemma tells executor to run a tool by emitting a **JSON “Action” block** as the *last* line of its response:

```json
Action: {
  "tool": "search_pubmed",
  "args": {
     "query": "adult refractory ketone ester epilepsy",
     "expand_mode": "semantic"
  }
}
```

Rules:

* Only one `Action:` per reply.
* Arguments **must** match the schema in `docs/70_tools.md`; extra keys cause executor error.
* If Gemma forgets to end with `Action:` the executor interprets the reply as “thoughts only” and the cycle ends without side effects.

---

## 4  Cycle Scheduler

```
while True:
    load goal_state
    if status == "stopped": break

    # 1. Retrieval / Expansion
    run any pending tool from previous cycle result

    # 2. Organisation
    recluster new+border (if split/merge last cycle)

    # 3. Gemma thinking
    prompt = make_prompt(cluster_metrics, sample_titles, handbook, goal_state)
    gemma_reply = call_gemma(prompt)
    parse_and_execute_Action(gemma_reply)

    # 4. Decision
    if success_condition_met:
        goal_state.status = "stopped"
    persist goal_state
    sleep(loop_delay)
```

* `loop_delay` default 5 seconds to avoid hammering APIs.
* Only one Gemma prompt per cycle keeps GPU util predictable.

---

## 5  Example Prompt Snippet (generated)

```
# Cluster 07e3a1dca9b4 – KD vs AED paediatric
Size: 146 | ERS: 0.32 | het_numeric: 0.54 | CVGR: 0.08
Unseen_upstream: 3  | Unseen_downstream: 72
5 sample titles:
• Ketogenic Diet in Drug-Resistant Childhood Epilepsy...
• Double-Blind Trial of 4:1 KD vs Placebo...
...

Handbook says Primary_goal = "seek_gap".
Success requires eligible_trials ≥ 6 & PRISMA_compliant false.
You may use tools: search_pubmed, run_pico, find_existing_sr, propose_alternative_pico, prisma_check.
Current eligible_trials: 5
Existing_SR_overlap: 0.21

<< write your reasoning >>
<< end with Action {...} >>
```

---

## 6  Gemma Behavioural Expectations

* **Chain-of-Thought length**: unbounded, but logs rotate; no penalty for verbosity.
* **Tool choice**: must align with whitelist else executor aborts cycle.
* **Pivoting**: Gemma sets `primary_goal="conduct_SR"` and fills `cluster_id` when it decides a gap is ready for a full review.
* **Stopping**: Gemma sets `status="stopped"` or executor auto-stops when `Stop_after` count reached.

---

## 7  Failure Handling

* Missing or malformed `Action` → executor logs `no_action` event, continues next cycle.
* Unknown tool → executor writes error to `history`, sets `status:"error"`, halts goal.
* `run_pico` cost guard: if cluster size > 800 abstracts, executor confirms with Gemma (“Are you sure?”) before batch call.

---

## 8  Extending Handbooks

* Adding a new **Primary\_goal** requires writing a success expression and possibly new tools.
* Handbook parsing is strict YAML front matter; syntax errors abort run at startup.

---

*This memo equips you to craft handbooks and understand how Gemma and the executor negotiate actions each cycle.  Next memo `70_tools.md` defines the heavy-tool API contracts Gemma can rely on.*
