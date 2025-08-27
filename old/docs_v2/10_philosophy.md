**Filename:** `docs/10_philosophy.md`
*(This memo captures every guiding principle, assumption, and non-negotiable rule that shapes all other modules. Nothing here should need to change unless we switch hardware or core LLMs.)*

---

# 10 – Architectural Philosophy & First Principles

> “Plumbing shouldn’t think and AI shouldn’t fetch URLs.”

Everything in this project follows that mantra.
Below we spell out the doctrines that inform design choices, thresholds, and coding style.
When a future change feels reasonable, measure it against these principles first.

---

## 1  Division of Labour

| Layer                                        | Responsibilities                                                                                                                                                                                            | Never Does                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Deterministic code** (“muscle & skeleton”) | • Download records from PubMed / Crossref<br>• Insert & dedup SQL rows<br>• Generate embeddings with Qwen<br>• Cluster with HDBSCAN<br>• Apply cosine maths, RAM checks, OOM fallbacks                      | Write English, decide clinical relevance |
| **Gemma 3 n** (“autonomous researcher”)      | • Read titles/abstracts and extracted fields<br>• Label, merge, split clusters<br>• Decide which heavy tool to run next<br>• Judge SR overlap, PRISMA compliance<br>• Mutate `goal_state.json` (pivot/stop) | Call HTTP APIs directly, handle SQL      |
| **Qwen 0.6 B** (“sensory cortex”)            | • Produce 1024-d vectors for any text, inc. instruction queries and PICO fields                                                                                                                             | Decide anything; it’s dumb math          |

---

## 2  Handbook-Driven Autonomy

*Every run is controlled by a Markdown (or Yaml) handbook.*
Sections:

```
Primary_goal: seek_gap | conduct_SR | ...
Seed_query:  "ketogenic diet AND epilepsy"
Success:      eligible_trials≥6 AND PRISMA_compliant==false
Tools_allowed: search_pubmed, run_pico, ...
Exploration:   true | false
```

* Why? Reproducibility and audit—changing behaviour never requires code edit, only handbook diff.
* Gemma sees the handbook at the top of every cycle; executor validates that a chosen tool is permitted.

---

## 3  Self-Mutating Goal Object

```json
goal_state.json
{
  "primary_goal": "seek_gap",
  "topic": "ketogenic diet epilepsy adults",
  "status": "in_progress",
  "cluster_id": null,
  "subgoals": []
}
```

* Gemma can rewrite anything except history.
* Example pivot: `"primary_goal":"seek_gap" → "conduct_SR"` once enough RCTs exist.
* Executor simply serialises the mutation; no logic second-guesses Gemma.

---

## 4  Clustering Doctrine – “Stable Buckets, Fluid Members”

1. First HDBSCAN pass yields provisional clusters.
2. **Gemma validates** each cluster:

   * writes a 10-word label + 50-word summary,
   * decides to keep / merge / split.
3. Accepted cluster gets:

   * **Frozen centroid** vector (mean of `[I|C|O|StudyDesign]` embeddings).
   * **Immutable ID** = `sha1(label + timestamp)[:12]`.
4. On later expansions:

   * New paper joins closest centroid **if** cosine ≤ `τ_assign`;
     `τ_assign = 95th percentile` of existing members’ distances.
   * Papers outside all centroids remain `unassigned` until next Gemma cycle.
5. Splits occur *only* when Gemma calls `split_cluster`; executor reruns HDBSCAN on that cluster only.
6. Merges occur analogously.

*Result:* textual names stay meaningful; graphs may grow without invalidating Gemma’s mental map.

---

## 5  Expansion Philosophy

### 5.1 Downstream (children) – “Semantic + Human Gate”

* Start with cosine threshold `τ0 = 0.60`.
* Batch boundary of 40 abstracts around τ; Gemma votes keep/discard.
* If Gemma keeps any → accept them, decrease threshold by δ = 0.05, loop.
* Stop when Gemma keeps zero or free RAM within 2 GB of limit.

### 5.2 Upstream (parents) – “Enough but Not Too Many”

* Count how often each parent DOI/PMID is cited inside the cluster.
* Select parents until cumulative citation share ≥ **α = 0.25**, but no more than **30**.
* Gemma reviews and may reject irrelevant reviews; if >80 % rejected halve α and retry once.

Downstream then ripples one hop from accepted seeds; Gemma can ask to switch expansion mode per cycle.

---

## 6  Embedding & Token Policies

* Qwen vectors are **FP16** mmapped; 10 M papers ≈ 20 GB.
* Token soft-cap per Gemma prompt **9 500 tokens**.
  Sampling algorithm drops oldest abstracts until cap met.
* Average 80 tokens/abstract → Gemma usually sees 60–110 abstracts per cluster.

---

## 7  Logging & Guard-Rails

* Full prompts **including chain-of-thought** saved as Markdown in `logs/`.
* Logs older than 30 days gzipped.
* JSON parsing error ⇒ send “FIX JSON ONLY”; retry once, then skip.
* OOM on embedding ⇒ halve batch size; if still fails, log & skip batch.
* `STOP` file in project root halts after current cycle (used for manual audit).

---

## 8  Why Sequential Cycle Interleaving, not Multithread

* Single GPU avoids vLLM contention.
* Each cycle ≤ 20 min; interleaving two goals gives human-perceived concurrency.
* Cluster IDs include goal hash to prevent collisions.
* Upgrade path: run separate processes when dual-GPU hardware available.

---

## 9  Non-Negotiables

* All heavy calls (**search\_pubmed**, **run\_pico**, etc.) *must* flow through tool registry.
* No Gemma self-reflection disable; keep COT for audit.
* Abstracts-only ingestion unless explicit legal clearance for PDFs.
* Every parameter lives in `config.toml`; *no magic numbers* in code.

---

### This memo defines the “constitution.”

Every other documentation file (data ingest, expansion, config) must align with these first principles.
