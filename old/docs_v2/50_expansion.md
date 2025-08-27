**Filename:** `docs/50_expansion.md`

---

# 50 – Expansion Engine: Bringing New Papers Into a Cluster

Expansion is the agent’s “eyes and legs.”
It has two distinct modes, each tuned for a different question:

| Mode                                                             | When Gemma Chooses It                                                          | Core Idea                                                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **Semantic Boundary Loop** (`expand_mode:"semantic"`)            | Gemma needs more *directly on-topic* primary studies to finish or judge an SR. | March outward in cosine space, ask Gemma to validate the frontier.                                |
| **Adaptive Upstream/Downstream Ripple** (`expand_mode:"ripple"`) | Gemma wants historical roots or follow-up syntheses.                           | Pull in frequent parents (upstream) and then chase children (downstream), all with Gemma filters. |

The executor implements both algorithms; Gemma selects the mode per cycle via tool arguments.

---

## 1  Common Preliminaries

```json
Action: {
  "tool": "search_pubmed",
  "args": {
     "query": "adult refractory epilepsy ketone ester",
     "expand_mode": "semantic" | "ripple"
  }
}
```

Executor fetches up to **2 000** candidate abstracts (`retmax` configurable) and embeds them with Qwen.

All accepted PMIDs go through the clustering intake pipe defined in memo 40.

---

## 2  Semantic Boundary Loop (Mode "semantic")

### 2.1  Parameters (config defaults)

```
tau_start     = 0.60   # initial cosine threshold
delta_tau     = 0.05   # step down per loop
batch_size    = 40     # titles+abstracts for Gemma
no_keep_limit = 0      # stop when Gemma keeps zero in a batch
```

### 2.2  Algorithm

```
accepted = []
threshold = tau_start
while True:
    pool = {cand | cos(Q_vec, E_cand) >= threshold} - accepted
    if not pool: break
    boundary = random_sample(pool, batch_size)
    Gemma boundary_prompt(boundary) -> keep_ids
    accepted += keep_ids
    if len(keep_ids) == no_keep_limit:
        break
    threshold -= delta_tau
```

* `Q_vec` = **centroid of the current cluster** (not the query text).
* Each loop shrinks the semantic circle; Gemma vetoes noise.
* When `threshold` drops below 0.30 or RAM guard trips, loop stops.

### 2.3  Why This Works

* High precision at start; recall improves gradually.
* Gemma boundary check keeps quality without inspecting thousands of abstracts.

---

## 3  Adaptive Upstream/Downstream Ripple (Mode "ripple")

### 3.1  Upstream Parent Selection

1. For cluster size `N`, count parent citations:

```
count(p) = |{child ∈ cluster : ref(child, p)}|
f(p)      = count(p)/N
```

2. Sort parents by `f(p)` descending.
3. Accept parents until **coverage Σ f(p) ≥ α** (`α = 0.25`) or **M = 30** parents collected.
4. Show titles to Gemma; if Gemma rejects ≥ 80 % → halve α and retry once.

Parents that survive are added to DB and set `seed_set`.

### 3.2  Downstream Evidence Chase

```
current = seed_set
while RAM_ok:
    children = {citer(p) for p in current} - known_papers
    E_child, cos = embed & cosine to cluster centroid
    keep = {child | cos >= tau_sem}          # tau_sem 0.45
    Gemma boundary_prompt(sample(keep,40)) -> keep_ids
    accepted += keep_ids
    current = keep_ids
    if len(keep_ids)==0: break
```

* No fan-out caps; Gemma decides when semantic drift begins.
* Loop ends when Gemma passes no IDs or RAM guard reached.

---

## 4  Integration with Goal Logic

* After expansion, executor **writes count of new accepted papers** to `cluster_manifest`; Gemma sees the delta next cycle.
* If new size ≥ `min_eligible_trials` (6 by default), Gemma may trigger `run_pico` or `prisma_check`.
* If Gemma sees `sr_overlap > 0.6` she can call `propose_alternative_pico` and pivot.

---

## 5  RAM Guard Details

* Guard threshold = `psutil.virtual_memory().available < 2*1024*1024*1024` (2 GB).
* When triggered inside either algorithm, executor stops adding further PMIDs and logs `"expansion_halt: RAM guard"`.

---

## 6  Tests

* **test\_semantic\_boundary\_shrinks** — ensure `threshold` monotonically decreases until stop.
* **test\_parent\_coverage** — after upstream selection `Σ f(p)` ≥ 0.25 or parent count == 30.
* **test\_ram\_guard** — simulate small RAM, loop exits early.

---

## 7  Tunable Scalars (duplicated in `90_config.md`)

```
alpha_upstream        = 0.25
max_parents           = 30
tau_start             = 0.60
delta_tau             = 0.05
tau_sem               = 0.45
boundary_batch_size   = 40
```

---

The expansion engine now fulfils both precision queries and exploratory ripples while handing the final relevance decision to Gemma.  Next memo `60_agent_handbook.md` describes how Gemma is instructed to invoke these expansion modes and other tools.
