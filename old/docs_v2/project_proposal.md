### **Project Proposal**

**Autonomous Literature-Discovery & Systematic-Review Agent**
*(Gemma 3n Reasoning + Qwen 0.6 B Retrieval, PubMed-scale, laptop-hardware)*

---

## 1  Introduction & Rationale

Systematic reviews (SRs) and meta-analyses are the gold standard for evidence synthesis, yet the manual workflow—formulating search strategies, screening abstracts, checking PRISMA items, assessing gaps—remains painfully slow.  Large-language models (LLMs) can now read, reason and plan, while cheap embedding models can project millions of abstracts into a coherent semantic space.
The project goal is to combine the two:

* **Gemma 3n (local)** supplies human-grade comprehension, planning, and adaptation—“the autonomous researcher”.
* **Qwen-0.6 B** produces dense vectors for **instruction-aware retrieval** and cheap similarity tests—“the sensory cortex”.
* Deterministic Python plumbing performs data ingestion, embedding, clustering, PICO extraction, graph walks—“the muscle & skeleton”.

The resulting agent should:

1. **Scout for untouched SR opportunities** (“review gaps”).
2. **Pivot goals** when prior SRs already exist, crafting alternative PICO themes automatically.
3. **Conduct a full PRISMA-compliant SR** when the evidence base is ready.
4. **Run end-to-end on a single high-end laptop** (i7-13700H, 32 GB RAM, RTX 4050 6 GB).

---

## 2  Guiding Design Principles

| Principle                                  | Implementation consequence                                                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **Separation of cognition vs. plumbing**   | Deterministic code owns all data movement; Gemma only reads language and chooses next tool.                                       |
| **Cluster scaffold, paper-level rigour**   | HDBSCAN groups papers for efficiency; Gemma can still operate per-paper (PICO, PRISMA) within any cluster.                        |
| **Open tool registry, not verb whitelist** | Any heavy or external call (PubMed API, batch PICO) is a tool. Gemma can chain tools freely; executor blocks only unknown names.  |
| **Self-mutating goal object**              | `goal_state.json` is rewritten by Gemma to pivot topics or goals without human intervention.                                      |
| **Context not budget-starved**             | Laptop handles ≈10 k tokens; sample 6-24 abstracts per cluster is affordable.                                                     |
| **Noise control via composite scoring**    | Candidate papers from PubMed are accepted only if semantic cosine + bibliographic coupling exceed a threshold—no hard count caps. |

---

## 3  High-Level Loop

```
┌──── Retrieval ─────┐  graph-walk + Gemma queries
└──── Organisation ──┘  HDBSCAN, metrics, Gemma merge/split/label
┌──── Processing ────┐  Gemma runs tools: ensure_completeness, run_pico, etc.
└──── Decision ──────┘  Gemma mutates goal_state or stops
(repeat)
```

*Everything Gemma does is dictated by a **handbook prompt** bundled with the goal.*

---

## 4  Modular Architecture

| #      | Module                   | Owner                           | Key I/O                                                                    | Extensibility                          |
| ------ | ------------------------ | ------------------------------- | -------------------------------------------------------------------------- | -------------------------------------- |
| **A**  | Data layer               | deterministic                   | `papers.db`, `G.pkl`, `E.npy`                                              | swap in full-text bucket later         |
| **B1** | Graph-walk expansion     | deterministic                   | adds refs+citers                                                           | depth configurable                     |
| **B2** | Semantic-query expansion | deterministic (Gemma-initiated) | PubMed → score(cos, coupling) → filtered IDs                               | adjust score weights/threshold         |
| **C**  | Organisation             | det. + Gemma                    | clusters, metrics, labels                                                  | drop-in new clustering (e.g. spectral) |
| **D**  | Tool registry            | det. + Gemma                    | `search_pubmed`, `run_pico`, `prisma_check`, `propose_alternative_pico`, … | register new tools anytime             |
| **E**  | Goal engine              | Gemma                           | `goal_state.json`                                                          | multiple simultaneous goals future     |
| **F**  | Logging & Guard-rails    | deterministic                   | full prompts + COT stored                                                  | OOM trimming, JSON fix-retry           |

---

## 5  Key Algorithms

### 5.1 Composite Relevance Score (Semantic Expansion)

```
cos_sem  = ⟨Q_vec,   E_cand⟩          # Qwen instruction embed
jac_refs = |refs∩seed| / |refs∪seed|   # bibliographic coupling
score    = 0.7·cos_sem + 0.3·jac_refs
accept if score ≥ τ   (e.g. 0.42)
```

Accepted seeds ripple ≤1 hop in citation graph until `growth_cap` reached.

### 5.2 Heterogeneity Fusion

```
het_numeric = mean(het_P, het_I, het_O)   # cosine variety
Het_final   = 0.5·Gemma_het + 0.5·het_numeric
```

Used only for ranking, never a gating criterion.

---

## 6  Tool Inventory (initial)

| Tool (heavy)                                                                      | Payload size  | When Gemma calls             |
| --------------------------------------------------------------------------------- | ------------- | ---------------------------- |
| `search_pubmed`                                                                   | API roundtrip | fill evidence gaps           |
| `run_pico`                                                                        | batch Gemma   | when cluster looks promising |
| `prisma_check`                                                                    | single Gemma  | after ≥ min\_eligible trials |
| `find_existing_sr`                                                                | API + cosine  | evaluate prior SR overlap    |
| `propose_alternative_pico`                                                        | Gemma plan    | pivot to new theme           |
| `score_candidates`                                                                | GPU batch     | part of expansion scoring    |
| *(light helpers)* `add_papers`, `compute_metrics`, `recluster` hidden from Gemma. |               |                              |

Adding e.g. `risk_bias_assess` later is a single registration.

---

## 7  Handbook Templates

### 7.1 Gap-Scouting Handbook (excerpt)

```markdown
Primary_goal: seek_gap
Seed_query: "ketogenic diet AND epilepsy"

Success:
  eligible_trials >= 6  AND  PRISMA_compliant == false

Tools_allowed:
  search_pubmed, score_candidates, add_papers,
  run_pico, find_existing_sr, propose_alternative_pico
```

### 7.2 Conduct-SR Handbook (excerpt)

```markdown
Primary_goal: conduct_SR
Cluster_id: 2            # inserted by previous run

ensure_completeness:
  min_eligible: 10
  exploration: true      # may search_pubmed

run_SR_pipeline:
  sequence: run_pico -> prisma_check -> summarise_findings
```

---

## 8  End-to-End Example Outcomes

1. **Topic reviewed already** → Gemma detects high SR overlap → generates new PICO on “MCT vs Classical diet adults/teens” → completes SR.
2. **Sparse evidence** → `ensure_completeness` loops semantic expansions until trials ≥ 6 → drafts gap dossier.
3. **Citation dead-end** → semantic query injects fresh adult trials → cluster revived.
4. **Data flood** → composite score thresholds admit only coherent seeds, preventing OOM.

---

## 9  Implementation Roadmap (module order)

1. **A** Data ingest + Qwen embeddings
2. **B1/B2** retrieval engines + scoring
3. **C** clustering, metrics, Gemma merge/split label
4. **D** tool executor + open registry
5. **E** goal\_state manager + handbook loader
6. **F** logging, guard-rails, diagnostics

Each module is unit-testable; integration smoke test uses 5 k-paper mini-corpus.

---

## 10  Risks & Mitigations

| Risk                             | Mitigation                                      |
| -------------------------------- | ----------------------------------------------- |
| Semantic drift from Qwen updates | version pin Qwen; re-embed on upgrade           |
| Gemma loops infinitely           | executor caps 1 000 tool calls / session        |
| API downtime (PubMed)            | local SQLite mirror snapshot fallback           |
| Hallucinated JSON                | one retry with “FIX JSON ONLY”, else skip cycle |
| Legal text ingestion             | abstracts-only; PDFs optional encrypted bucket  |

---

## 11  Future Extensions

* **GraphSAGE-based citation intent** edges (support, contrast).
* **Risk-of-bias LLM tool** for SR drafts.
* **Interactive dashboard**—clusters visualised, Gemma decisions replayed.
* **Multiple concurrent goals** with shared literature cache.

---

### **Conclusion**

This modular plan gives you:

* **Full autonomy**—Gemma can pivot topics and goals when evidence dictates.
* **No hidden coupling**—each module swaps cleanly, handbook text rewires behaviour.
* **Predictable resource use**—large semantic jumps are pruned by composite scoring; GPU memory guarded.
* **Laptop viability**—max context ≈10 k tokens, embedding batches throttled.

With the six core modules in place, you can incrementally bolt on smarter embeddings, deeper bias checks, or richer graphs **without reopening architectural debates**.
