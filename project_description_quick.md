# Autonomous Literature Explorer — Final System Blueprint + Multi-PoC Plan (no code)

Below is a **full, systems-level specification** of the architecture we’ve co-designed, written to be handed directly to an implementation team (or another AI) without ambiguity. After the blueprint, you’ll find a suite of **Proof-of-Concepts (PoCs)** that exercise the system in independent slices. This deliberately avoids any UI and assumes **fully autonomous operation**, DOI-first ingestion, PubMed/Crossref as sources, and your hardware constraints.

---

## 1) Scope and goals

We are automating exploratory literature intelligence for biomedical questions, with the ability to:

* discover and refine a **research question** (PICO) from recent literature;
* **find and promote primary studies** into a curated **Sentinel Set** (our evolving “included studies” surrogate);
* **expand via the citation graph** with bounded ripple;
* decide **whether a systematic review (SR) already exists** that truly covers the question;
* optionally trigger **update-hunt** if an SR exists but misses new key primaries.

Design principles:

* **LLM for language & policy** (Gemma 3n, \~2B active by default): PICO gate, SR verdict, plan selection, identity arbitration.
* **Deterministic math** for measurement & safety: retrieval, ranking, similarity, overlap, gating thresholds.
* **Short episodes** with explicit quotas so the agent learns and adapts rapidly.
* **No human-in-the-loop** (for PoC), **no UI**; fully autonomous.

---

## 2) High-level architecture (3 cooperating layers)

### (A) Perception & Measurement

* **Hybrid retrieval:** BM25 (keyword) + dense (Qwen embeddings).
* **Graph exploration:** ripple **upstream** (references) and **downstream** (citers).
* **Embeddings:** abstract embedding + 5 **facet** embeddings for P/I/C/O/S.
* **Signals:** cosine similarities, BM25 z-scores, structural overlaps, discovery curves.

### (B) Cognition & Policy (Gemma)

* **Mode selection & plan writing** for each episode, based on a compact **instrument panel** (state digest).
* **PICO gate**: binary on-PICO primary? (batched, JSON, tiny).
* **SR verdict**: “SR\_exists / SR\_partial / SR\_outdated / No\_SR”, with **label-quoting** justification.
* **Identity arbitration** for near-duplicate questions.
* **Query rewrites** (PRF) and gate label changes.

### (C) Memory & Governance

* **Agenda** of Research Questions (RQ) with canonical one-line PICO, facet vectors, **Focus Set**, **Sentinel Set**, and episode logs.
* **Epochs**: reopen an RQ later under the same identity.
* **Persistence**: papers, embeddings, edges, agenda, sentinels, episodes, SR candidates, decisions.

---

## 3) Data model & identifiers (strict for PoC)

* **ID policy:** **DOI is mandatory** for every ingested item. PMID is mapped where present (build a DOI↔PMID cache).
* **Sources:** PubMed (citations, metadata) and Crossref (references, DOIs). No fuzzy title matching in PoC.
* **Retractions:** at **Sentinel promotion** time only—if PubMed indicates **Retracted Publication** (or title begins “\[Retracted]”), **deny promotion**.
* **Preprint→journal linkage (minimal):** if Crossref/PubMed explicitly declares a relationship (published-as / is-preprint-of), keep the **journal DOI** as canonical; otherwise do nothing.

---

## 4) Retrieval & ranking (hybrid)

### 4.1 BM25 (lexical)

* Index titles+abstracts.
* Normalize BM25 per batch to **z-scores**:
  `z = (bm25 − mean_pool)/std_pool` for the candidate pool in this episode.

### 4.2 Dense (vector)

* Embed queries with Qwen (instructed for scientific abstract retrieval).
* Retrieve top-k by cosine.

### 4.3 Hybrid scoring

Example linear combiner:

```
score = 0.35 * cosine_to_focus_centroid
      + 0.25 * BM25_z
      + 0.20 * recency_scaled              // e.g., 1.0 newest → 0 older
      + 0.15 * type_boost                  // RCT boost in Primary-scout; SR boost in SR-hunt
      + 0.05 * novelty_distance            // distance from accepted set centroid
```

Take top-K to screening.

### 4.4 Pseudo-relevance feedback (PRF)

* Extract discriminative terms (drug names, comparator aliases, outcomes) from **recently accepted** abstracts.
* Add to **Boolean** and **dense** queries next episode; prune terms correlated with false positives.

---

## 5) Vector index (ANN) — HNSW on CPU

* Use **HNSW** (e.g., FAISS IndexHNSWFlat or hnswlib) on CPU to serve dense retrieval.
* Typical params: `M=32, efConstruction=200, efSearch=128–256`.
* Rationale: fits your 32 GB RAM; keeps the GPU for Gemma; sub-ms to few-ms per query up to \~10⁶ vectors with high recall.

---

## 6) Graph ripple with explosion control

### 6.1 Credits & quotas

* Each episode has a **global ripple quota** (e.g., admit ≤ M nodes).
* Each **source node** gets a small **credit** budget for references/citers (primaries > reviews > guidelines).

### 6.2 Gates (must pass to be admitted)

* **Semantic gate (adaptive):** cosine ≥ **percentile** of similarity among **recently accepted** items. Labels:

  * `strict` ≈ 60th percentile,
  * `normal` ≈ 40th,
  * `lenient` ≈ 20th.
    Gemma requests the label; the orchestrator computes the numeric threshold.

* **Structural gate (simple and robust):** require overlap with what we already trust.
  Define a trusted set **U** (PoC: `U = Sentinels ∪ Focus`).

  * Upstream support: `support_up(v) = |Ref(v) ∩ U| / |Ref(v)|`.
  * Downstream support: `support_down(v) = |Citers(v) ∩ U| / |Citers(v)|`.
  * Admit if `support_up ≥ α` OR `support_down ≥ β` (PoC defaults α=β=0.15).
    **Recency prior:** allow recent high-semantic items even if support is low.

* **Degree caps:** hard cap per source (e.g., process ≤100 references or ≤100 citers per node per episode).

* **Innovation budget:** reserve \~10% of ripple quota for **bridge** candidates (high novelty vs centroid or high semantic betweenness) that fail structural support—prevents ossification.

*(Optional for V2: size-aware **hypergeometric** p-value gate; time-window attenuation of old refs.)*

### 6.3 Priority after gating

```
priority(v) =
  λ1 * sim_focus(v)
+ λ2 * support_up(v)
+ λ3 * support_down(v)
+ λ4 * bridge_score(v)
+ λ5 * recency(v)
- λ6 * hub_penalty(v)    // e.g., degree discount 1/sqrt(outdegree)
```

Admit in decreasing priority until quotas are filled.

### 6.4 Hub attenuation in metrics

When computing walk-based influence (RWR/PageRank) or selecting landmarks, scale each outgoing edge by **1/√outdegree** and a **type factor** (primary=1.0, review=0.7, guideline=0.2).

---

## 7) Focus & Sentinel sets

### 7.1 Focus set (topic spine)

* \~20–50 papers most central **now** (mix of key primaries + on-point reviews).
* Used to compute centroids and stabilize retrieval/gating.

### 7.2 Sentinel set (our “included studies” surrogate)

**Promotion pipeline (per episode):**

1. **Candidate pool** from ranked retrieval + ripple.
2. **Cheap filters:** PubMed PublicationType (if available), date windows, NOT animals; **skip retracted**.
3. **PICO gate (Gemma, batched, tiny)** on **ambiguous** items only (metadata-clear RCTs can bypass):

   * `is_primary_on_this_PICO: yes|no|uncertain`
   * `why_5_words: string`
   * `design_hint: RCT|cohort|case-control|other`
4. **Ranking for promotion:** facet-weighted proximity (I,C heavy), recency, design preference (RCT first), novelty vs current sentinels.
5. **Admit top-m** (e.g., 5–10) to Sentinel **S** this episode. Keep “near-accepts” in a queue.
6. **Maintenance:** if a later, better report supersedes a smaller duplicate, demote; **freeze S** once coverage stabilizes for this epoch.

---

## 8) Exhaustiveness audits (stop when “enough”)

We mark S “exhaustive for the epoch” when all three agree:

* **A) Discovery-curve saturation:** slope (new sentinels per **reads**) below ε for consecutive episodes; not improving after PRF.
* **B) Capture–recapture:** overlap of primaries found via retrieval vs ripple indicates few unseen remain.
* **C) SR-reference audit:** from top SR candidates, references that pass the PICO gate but are **not in S** are promoted or rejected; iterate until the **missing-from-S** trickle is trivial or out-of-scope.

*(For PoC, A + C are sufficient; B is a nice-to-have if easy to compute.)*

---

## 9) SR coverage & verdict

### 9.1 Evidence brief (compiled by measurement)

For each candidate SR `r`:

* **coverage** label over S: bucket of `|S ∩ C_r| / |S|` where `C_r` = SR references that pass the PICO gate (labels: `very_high | high | medium | low | unknown`).
* **P/I/C/O match**: `strong | moderate | weak` (facet cosine).
* **recency**: `current (≤3y) | recent (≤5y) | stale`.
* **guideline\_like**: `yes|no`.
* **missing\_sentinels\_DOIs**: list (if any).

### 9.2 LLM verdict (Gemma) with label-quoting discipline

* Output:

  * `decision`: `SR_exists | SR_partial | SR_outdated | No_SR`
  * `best_candidate`: DOI/PMID or null
  * `justification`: two bullets that **quote** the evidence labels verbatim
* **Validator** enforces schema + quoted labels; one auto-repair retry; rare escalation.

---

## 10) Update-hunt “greediness”

Compute an **Update Trigger** (U) from:

* **age\_sr**, **miss\_frac** (fraction of sentinels missing), **postdate\_frac** (fraction of sentinels newer than SR’s search window), **scope\_penalty** (any facet ≤ moderate), **new\_methods** (e.g., first large RCT appears).

Select policy:

* **strict / balanced / greedy** threshold on U.
  If triggered: enter **update-hunt** mode (search **after** SR search date; ripple from new sentinels; produce a delta list of new primaries).

---

## 11) Identity & epochs

* **Semantic key**: weighted facet cosine (I,C highest).
* **Evidence key**: sentinel Jaccard (once S exists).
* If high semantic + (unknown or high evidence) → **same RQ**; if borderline → **Gemma** labels `same | broader | narrower | different` and **names the facet**; cache decisions.
* **Epochs** reopen the same RQ later; do not clone.

---

## 12) Rapid Focus (bootstrap a new area)

* Pull last **12–18 months** via hybrid retrieval.
* Compute **term velocity** for top 1–2k n-grams (slope over time; lightweight EMA smoothing).
* One **micro-clustering** pass on embeddings; rank clusters by **velocity-weighted** salience.
* Gemma proposes 2–3 **one-line** PICO candidates (“why now” uses recency/velocity); select one automatically by salience if headless; start **Primary-scout**.

---

## 13) Bandit auto-tuning (simple, no ML training)

* Choose discrete **arms** (e.g., semantic gate = lenient/normal/strict).
* **Payoff:** **new sentinels per N reads** (e.g., per 200 abstracts screened).
* Episodes 1–3 explore each arm; from episode 4 use **ε-greedy** (e.g., 80–90% best arm; 10–20% exploration).
* Apply similarly to structural threshold α or ripple refs vs citers split if desired.

---

## 14) LLM prompt schemas (spec sheets)

### 14.1 PICO-gate (batched)

**System**: You classify biomedical abstracts. Output strict JSON per item. No explanations.

**Per item input**: `{title, abstract, PICO_target: {P,I,C,O,S}}`

**Output JSON per item**:

```json
{
  "paper_id": "<doi-or-pmid>",
  "is_primary_on_PICO": "yes|no|uncertain",
  "design_hint": "RCT|cohort|case-control|other|unknown",
  "why_5_words": "<short reason>"
}
```

**Validation rules**:

* `is_primary_on_PICO` in enum.
* If `is_primary_on_PICO="yes"`, then `design_hint` ≠ "unknown".
* `why_5_words` length ≤ 7 tokens.

### 14.2 SR verdict

**System**: You render an SR coverage verdict **only** from the provided labels.

**Input**: `{PICO_target, evidence_briefs:[{candidate_id, coverage, I_match, C_match, O_match, recency, guideline_like, missing_sentinels_DOIs[]}]}`

**Output**:

```json
{
  "decision": "SR_exists|SR_partial|SR_outdated|No_SR",
  "best_candidate": "<doi-or-pmid-or-null>",
  "justification": [
    "coverage=<label>; I-match=<label>; C-match=<label>; recency=<label>",
    "<optional second bullet quoting labels>"
  ]
}
```

**Validation rules**:

* `decision` in enum; if `SR_exists|SR_partial|SR_outdated` then `best_candidate` ≠ null.
* Every bullet must **quote** at least two labels verbatim (`coverage=...`, `I-match=...`, `C-match=...`, `recency=...`).

---

## 15) Persistence (minimal tables)

* `papers(id_doi PK, pmid, title, abstract, year, pub_types[], flags{retracted})`
* `embeddings(id_doi FK, model_ver, vect_abstract, vect_P, vect_I, vect_C, vect_O, vect_S)`
* `edges_cites(src_doi, dst_doi)` and `edges_citers(dst_doi, src_doi)` if separated by source
* `agenda(rq_id PK, pico_line, facet_vecs, epoch, status)`
* `sentinels(rq_id, id_doi)` plus `near_accepts(rq_id, id_doi)`
* `episodes(rq_id, episode_id, mode, plan_json, caps, metrics_json)`
* `sr_candidates(rq_id, id_doi, evidence_brief_json, verdict_json)`
* `decisions_identity(rq_id_a, rq_id_b, label, reason, timestamp)`

Caches:

* `doi_pmid_map(doi→pmid, pmid→doi)`
* ANN index on embeddings (on disk).

---

## 16) Runtime & resource guidance (RTX 4050 6 GB)

* **Embeddings** on CPU; HNSW index in RAM; \~few GB for \~10⁵–10⁶ papers.
* **Gemma** GGUF quantized (int4/int8) in LM Studio; **batch** PICO-gate 16–32 items; reuse session; short max tokens.
* **Per-episode quotas** (tune later):

  * retrieve top-K=600 lexical + 400 dense;
  * read ≤ 200 abstracts (screening) per episode;
  * admit ≤ 10 sentinels; ripple admit ≤ 300 items.

---

# PoC Suite (no code) — independent slices with clear inputs/outputs/metrics

Each PoC is **headless**, autonomous, and logs JSON artifacts. No human action.

---

## PoC-0: Infrastructure smoke test

**Goal:** Confirm data flow, IDs, embeddings, ANN, and BM25 work end-to-end.

* **Inputs:** Small corpus (e.g., 50k PubMed+Crossref records with DOIs), your target PICO text.
* **Procedure:** Index BM25; build HNSW; embed; run one hybrid query; verify DOI↔PMID mapping; test retrieve-by-vector.
* **Outputs:** retrieval JSON; top-k with BM25 z + cosine; timings; memory usage.
* **Pass if:** latency per query acceptable (<100 ms dense); indices load correctly; IDs resolve.

---

## PoC-1: Retrieval + ranking quality

**Goal:** Show hybrid score improves top-k over BM25 or dense alone.

* **Inputs:** Topic corpus; 5–10 known on-topic DOIs (seed list for evaluation only).
* **Procedure:** Run BM25 only, dense only, hybrid. Compute hit\@k (does seed list appear in top-k?).
* **Metric:** Relative improvement of hybrid vs each baseline.
* **Pass if:** hybrid beats both for hit\@k at k ∈ {50,100,200}.

---

## PoC-2: PICO gate (LLM) throughput & precision

**Goal:** Validate batched classification speed and precision for ambiguous cases.

* **Inputs:** 500 abstracts (mixed types).
* **Procedure:** Run batched PICO-gate; log decisions, design hints, reasons; measure throughput.
* **Metrics:** abstracts/sec; % of items classified as `yes` that match PublicationType=RCT/cohort when present (“silver” precision).
* **Pass if:** ≥ 85% silver precision; ≥ 50 abstracts/sec on your setup (tunable).

---

## PoC-3: Sentinel growth & Audit A/C

**Goal:** Show Sentinel Set grows and saturates; SR-reference audit reduces missing items.

* **Inputs:** One RQ; 5 consecutive episodes with quotas.
* **Procedure:** Run the core loop; log #sentinels per episode, “new per 200 reads”; run SR-audit each episode (promote missing-from-S if on-PICO).
* **Metrics:** discovery-curve slope; missing-from-S trend.
* **Pass if:** slope falls toward ε and SR-audit residual approaches zero by episode 4–5.

---

## PoC-4: Ripple gating ablation

**Goal:** Quantify effect of gates on explosion & relevance.

* **Inputs:** Same RQ as PoC-3.
* **Procedure:** Episodes with:

  * (i) no gates (degree caps only),
  * (ii) semantic only,
  * (iii) semantic + structural (α=0.15),
  * (iv) + innovation budget (10%).
* **Metrics:** admitted per episode; acceptance (kept/seen); % off-topic (heuristic); # new sentinels per 200 reads; wall-clock.
* **Pass if:** (iii) or (iv) delivers best **new-sentinels/reads** with bounded admitted count (no blow-ups).

---

## PoC-5: SR coverage & verdict

**Goal:** Demonstrate evidence-brief → LLM verdict mechanics.

* **Inputs:** One RQ with \~10–20 sentinels; 3–5 SR candidates; their PICO-gated reference lists (from Crossref/PubMed).
* **Procedure:** Compute coverage buckets; P/I/C/O labels; recency; run LLM verdict; validate JSON.
* **Artifacts:** evidence briefs; verdict JSON with **quoted** labels.
* **Pass if:** JSON always valid (≤1 repair retry), verdict stable across repeated runs, and intuitively consistent with labels.

---

## PoC-6: Update-hunt “greediness”

**Goal:** Verify Update Trigger switches behavior.

* **Inputs:** One RQ with an SR candidate; add 2 “new” sentinels post SR date.
* **Procedure:** Compute U under strict/balanced/greedy; observe whether update-hunt is triggered.
* **Pass if:** policy toggles behavior as designed.

---

## PoC-7: Rapid Focus with term velocity

**Goal:** Show velocity prioritizes hot clusters for PICO seeding.

* **Inputs:** Recent (12–18 months) corpus for a broad area.
* **Procedure:** Compute top n-grams; velocities; cluster abstracts once; rank clusters by velocity-weighted salience; have Gemma propose 2–3 PICOs; auto-select top; start one **Primary-scout** episode.
* **Metrics:** acceptance rate in episode 1; #sentinels admitted; comparison vs random cluster seed.
* **Pass if:** velocity-ranked seed yields higher acceptance and sentinel yield than a random seed.

---

## PoC-8: Identity & epochs

**Goal:** Validate same/broader/narrower/different arbitration.

* **Inputs:** 6 synthetic PICO lines (pairs with small controlled facet changes).
* **Procedure:** Compute facet similarities; ask Gemma for label; cache decisions.
* **Metrics:** sanity: high I/C similarity → “same”; change comparator → “different/broader/narrower”.
* **Pass if:** outputs align with controlled facet edits and are consistent across runs.

---

## PoC-9: Contradiction-probe micro

**Goal:** Show contradiction trigger and adjudication work in miniature.

* **Inputs:** \~12 primaries with claim stubs (half “increase”, half “no effect”) under nearly same PICO.
* **Procedure:** Compute contradiction index; trigger probe; run a narrow retrieval; ask Gemma to adjudicate (name differing facet).
* **Pass if:** adjudication cites facet differences or confirms genuine contradiction with a short rationale.

---

## PoC-10: Bandit auto-tuning (ε-greedy)

**Goal:** Improve gate choice based on **new sentinels per 200 reads**.

* **Inputs:** One RQ; 6 episodes.
* **Procedure:** Episodes 1–3 explore lenient/normal/strict; episodes 4–6 pick best arm with ε=0.2 exploration.
* **Metric:** average S/200reads episodes 4–6 > episodes 1–3 best single arm baseline (or matches with less variance).
* **Pass if:** bandit doesn’t degrade performance and usually improves or stabilizes S/reads.

---

## PoC-11: Performance profile (RTX 4050)

**Goal:** Establish throughput ceilings.

* **Inputs:** A run of 5 episodes on one RQ.
* **Measurements:** per stage wall-clock, abstracts/sec in LLM gate, tokens per episode, HNSW query latency, memory footprint, GPU VRAM.
* **Pass if:** per-episode runtime within your tolerance; no OOM; throughput stable.

---

## Closing notes (PoC operating rules)

* **Autonomous only**: no UI, no manual overrides.
* **IDs strict**: **DOI required**; use PubMed only as a DOI/metadata partner; skip items lacking DOI in PoC.
* **Safety minimalism**: sanitize inputs lightly (strip control chars); validator for JSON outputs; no chain-of-thought.
* **Primary metric for tuning**: **new sentinels per 200 reads** (S/reads). Secondary telemetry: wall-clock.

This gives you both the **complete architectural reference** and a **concrete PoC test plan** that another AI (or team) can code against. When these PoCs pass, we’ll have hard evidence that the autonomous agent: (i) learns within episodes, (ii) grows and saturates a Sentinel Set, (iii) renders SR decisions from labeled evidence, and (iv) stays stable and fast on your hardware—exactly what we need before scaling features in V2.
