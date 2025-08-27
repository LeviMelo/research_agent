**Filename:** `docs/00_overview.md`

---

# Autonomous Literature-Discovery & Systematic-Review Agent

*(Gemma 3 n LLM + Qwen 0.6 B embeddings — single-laptop deployment)*

---

## 1  Why This Project Exists

Traditional systematic reviews (SRs) require weeks of manual database queries, duplicate screening, and PRISMA compliance checks. By fusing:

* **Gemma 3 n** — a locally-running 7-B model that can read, reason, and plan,
* **Qwen 0.6 B** — a GPU-friendly sentence-embedding model that turns every abstract into a 1024-d vector, and
* **deterministic Python plumbing** — for downloading PubMed/Crossref records, building a citation graph, and clustering papers,

we can create an *autonomous researcher* that:

1. **Hunts for evidence gaps** (fields with trials but no up-to-date SR).
2. **Pivots automatically** if it discovers a prior SR already covers the intended topic.
3. **Completes a PRISMA-ready SR draft** when enough trials accumulate.
4. **Runs entirely on commodity hardware** (i7-13700H + RTX 4050 6 GB, 32 GB RAM).

---

## 2  Laptop Hardware Envelope

| Component | Spec                                         |
| --------- | -------------------------------------------- |
| CPU       | 14-core Intel i7-13700H                      |
| RAM       | 32 GB DDR5                                   |
| GPU       | NVIDIA RTX 4050 Laptop (6 GB VRAM)           |
| Storage   | ≥ 1 TB NVMe (supports mmap of 20 GB vectors) |

*Context window*: Gemma comfortably handles **≈ 10 000 tokens** at 20-30 tok/s on this GPU.

---

## 3  High-Level Control Loop

```
┌────────────────── Retrieval ────────────────┐
│ 1. Gemma writes semantic PubMed query OR    │
│    asks for citation ripple                 │
│ 2. Deterministic code fetches papers,       │
│    adds embeddings, updates graph           │
└──────────────────────────────────────────────┘
┌────────────────── Organisation ─────────────┐
│ 3. HDBSCAN clusters embeddings              │
│ 4. Gemma labels / merges / splits clusters  │
└──────────────────────────────────────────────┘
┌────────────────── Processing ───────────────┐
│ 5. Gemma chooses tools: run_pico,           │
│    find_existing_sr, ensure_completeness…   │
│ 6. Deterministic executors run heavy calls  │
└──────────────────────────────────────────────┘
┌────────────────── Decision ────────────────┐
│ 7. Gemma mutates goal_state.json:          │
│      • continue   • pivot   • stop         │
└─────────────────────────────────────────────┘
(repeat until stop condition)

All parameters (cosine thresholds, batch sizes) live in **config.toml**; no constants are hard-coded.
```

---

## 4  Key Architectural Commitments

* **Cognition vs Plumbing** – Gemma never touches raw SQL or APIs; it only reads language and issues tool calls.
* **Frozen Centroids** – Once Gemma approves a cluster, its semantic centre and ID stay fixed; new papers join by cosine proximity.
* **Adaptive Expansion** – Downstream growth uses a decreasing cosine threshold with Gemma boundary checks; upstream uses reference-frequency + Gemma filter.
* **Self-Mutating Goals** – `goal_state.json` lets Gemma pivot from “conduct\_SR” to “seek\_gap” without human prompts.
* **Sequential Interleaving** – Multiple goals run in alternating cycles—no multithreading, no model contention.

---

## 5  Module Map (detailed in later memos)

| ID | Module         | Core Idea                             |
| -- | -------------- | ------------------------------------- |
| A  | Data Ingest    | PubMed + Crossref merge into SQLite   |
| B  | Expansion      | Semantic boundary + adaptive ripple   |
| C  | Organisation   | HDBSCAN → frozen centroids + metrics  |
| D  | Tool Registry  | Heavy operations Gemma can call       |
| E  | Goal Engine    | Mutable goal\_state + handbook prompt |
| F  | Logging/Guards | JSON retry, STOP flag, RAM watchdog   |

---

## 6  Success Criteria

* **Gap Scouting** – agent outputs Markdown dossier for one cluster with ≥ 6 eligible RCTs and *no* compliant prior SR.
* **Full SR** – agent produces PRISMA-check-passed draft (PICO table + abstract synthesis).
* **Runtime** – stays within 6 GB VRAM and 28 GB RAM; single cycle ≤ 20 minutes on laptop.

---

*This overview links to all deeper memos (`10_philosophy.md`, `20_data_ingest.md`, …) which specify algorithms, thresholds, and file layouts.*
