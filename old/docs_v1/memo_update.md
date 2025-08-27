### **Final Architecture Memo — Autonomous Literature-Discovery & SR Agent**

*(All structural decisions locked; ready to move on to coding details)*

---

## 1 Core Philosophy

1. **Deterministic plumbing** handles data ingest, embeddings, clustering, graph walks.
2. **Gemma 3 n** supplies all language understanding, strategic planning, filtering, and goal pivots.
3. **Qwen 0.6 B** provides instruction-aware sentence embeddings used everywhere cosine similarity is required.
4. **Single handbook prompt** per goal explains tools and success criteria; Gemma can mutate `goal_state.json` to pivot topics or switch from gap-scouting to full SR.
5. **Sequential cycles** (no true concurrency) but multiple goals are supported by interleaving cycles; each cycle touches only one goal.

---

## 2 Directory Layout

```
project_root/
  db/        papers.db, ref.sqlite
  emb/       E.npy, p.npy, i.npy, c.npy, o.npy, m.npy
  cache/     crossref_json/, pubmed_xml/
  logs/      YYYY-MM-DD_cycleN.md   (full prompts + COT)
  outputs/   dossiers/, sr_drafts/
  artifacts/ YYYYMMDD/  (weekly frozen snapshots)
  config.toml
  handbooks/ *.md
```

---

## 3 Static Extraction Schema (Gemma JSON per abstract)

| Key         | Contents (string)        | Embedding file |
| ----------- | ------------------------ | -------------- |
| P           | population incl. age & N | p.npy          |
| I           | intervention incl. dose  | i.npy          |
| C           | comparator               | c.npy          |
| O           | primary outcome          | o.npy          |
| StudyDesign | “RCT”, “cohort”…         | m.npy          |

Missing fields → empty string (zero vector).

---

## 4 Citation Graph Construction

1. **PubMed**: basic metadata + citing links among PMIDs.
2. **Crossref**: reference lists (DOI→DOI).  Dedup by mapping DOI→PMID.
3. Insert edges into `ref(src,dst)` with `INSERT OR IGNORE`.
4. Chronology inconsistencies tolerated (DAG not required).

---

## 5 Expansion Algorithms

* **Upstream (parents)**

  * Compute parent citation frequency `f(p)`.
  * Accept parents until cumulative `Σ f(p)` ≥ **α = 0.25** or **M = 30** parents, whichever first.
  * Pass titles to Gemma for semantic rejection; retry with α / 2 if Gemma rejects > 80 %.
* **Downstream (children)**

  * **Method C**: semantic cosine sort; start τ₀ = 0.60, decrease by δ = 0.05.
  * Boundary batch of 40 titles sent to Gemma each step; keep adding until Gemma returns zero relevant.

Both directions stop when free RAM < 2 GB.

---

## 6 Clustering & Stability

* **HDBSCAN** over `[I|C|O|StudyDesign]` embeddings.
* **First approval** freezes cluster ID (SHA-1 of label+timestamp) and centroid.
* New papers join nearest centroid if cosine ≥ **τ\_assign = 0.20**.
* Gemma may call `split_cluster` or `merge_clusters`; recluster only inside affected clusters.

---

## 7 Tool Registry (heavy calls only)

```
search_pubmed        -> JSON hits
embed_text           -> 1024-d vector(s)
run_pico             -> PICO JSONL
prisma_check         -> PRISMA compliance JSON
find_existing_sr     -> {overlap: float, ids:[…]}
propose_alternative_pico -> new PICO string
```

Executor queues calls; model contention solved by sequential cycles.

---

## 8 Goal-State Autonomy

```json
{
  "primary_goal": "seek_gap" | "conduct_SR",
  "topic": "string",
  "cluster_id": 17,
  "status": "in_progress" | "stopped",
  "subgoals": []
}
```

Gemma may edit any field except history; executor persists per cycle.

---

## 9 Adaptive Thresholds

* `τ_assign` and semantic τ in expansion **derived per cluster** from P95 of intra-cluster cosine distribution; default fallbacks 0.20 / 0.60.
* All thresholds stored in `config.toml`; logged per cycle for audit.

---

## 10 Guard-Rails & Logging

* Full Gemma chain-of-thought retained; logs rotate after 30 days (gzip).
* JSON parse error → retry once with “FIX JSON ONLY”.
* API failures → fallback to local PubMed baseline after 5 retries.
* `STOP` file in root pauses loop after current cycle.

---

## 11 Outstanding Scalars (all accepted as defaults)

```
α (upstream coverage)        = 0.25
M (max parents)              = 30
τ0 / δ (semantic expansion)  = 0.60 / 0.05
boundary_batch_size          = 40
τ_assign                     = adaptive (fallback 0.20)
min_eligible_trials          = 6
token_soft_cap               = 9500
```

You can change them later in config without code edits.

---