### **Technical Memo — Revision 2**

**Subject:** Embedding Layers, Sampling Policy, and Rich-Field Extraction
**Date:** *\[insert today]*

---

## 1  Embedding & Similarity Layers  (Updated)

| Vector set          | Generator                                  | Dim       | File                | Notes                                      |
| ------------------- | ------------------------------------------ | --------- | ------------------- | ------------------------------------------ |
| **E (semantic)**    | Qwen-0.6 B on `"Title · Abstract"`         | 1024 FP16 | `E.npy` (mmapped)   | Baseline content signal                    |
| **Z (context)**     | **GraphSAGE, 1-hop** on citation graph     | 256 FP16  | `Z.npy`             | Captures network position (up/down-stream) |
| **P / I / C / O**   | Qwen on Gemma-extracted fields (4 vectors) | 4 × 1024  | `p.npy` … `o.npy`   | Only after first PICO screen               |
| **M (methodology)** | Qwen on Gemma field `"StudyDesign"`        | 1024      | `m.npy`             | RCT vs cohort vs case-series               |
| **Centroids**       | mean of any chosen vector set              | —         | in cluster manifest | Recomputed each cycle                      |

### Why GraphSAGE is included

*Z* provides extra orthogonal evidence when semantic text is noisy (e.g. methods papers heavily cited).

### Why Comparator **C** is now explicit

Comparator often flips effect direction; needed for split/merge logic and heterogeneity.

### Extensible Rich Field Extraction

Gemma extraction schema (all strings):

```json
{
 "P": "...",
 "I": "...",
 "C": "...",
 "O": "...",
 "StudyDesign": "randomised double-blind",
 "PopulationAge": "children 2-12",
 "SampleSize": "47"
}
```

You can add new keys later—simply create another embedding matrix (e.g. `d.npy` for dose description); clustering formula can switch to concatenating `[I|C|O|M]` without architectural change.

---

## 2  Composite Similarity (E, Z, PICO, Methodology)

When all vectors exist:

```
sim = 0.40·cos(Q, E)  +
      0.15·cos(Q, Z)  +
      0.25·cos(Q, I⊕C⊕O) +
      0.20·cos(Q, M)
```

Weights are config; fall back gracefully if a component missing.

---

## 3  Dynamic Cluster Sampling Policy (Token-aware)

```python
MAX_TOK = 9500              # soft cap
tok_per_abs = 80            # empirical avg title+abstract
m = min(  max(6, floor(0.15*size)),  size )
while m * tok_per_abs > MAX_TOK:
        m -= 1              # trim until within token budget
```

*On large clusters Gemma may see 60–80 abstracts if they’re short; on small clusters it always sees all.*

---

## 4  Gemma Extraction Consistency Guard

After Gemma returns JSON for a paper:

* deterministic code **validates schema keys**; missing keys → empty string, placeholder vector = zero.
* titles with identical P/I/O but conflicting StudyDesign raise a flag for manual audit (logged).

---

## 5  Search-Expansion Scoring (with Graph & Methodology)

```
cos_sem  = ⟨Q,   E⟩
cos_ctx  = ⟨Q_Z, Z⟩                  # Q_Z = Qwen("context of "+query)
cos_pico = ⟨Q, I⊕C⊕O⟩
cos_meth = ⟨Q, M⟩
jac_refs = bibliographic coupling     # as clarified

score = 0.35·cos_sem + 0.15·cos_ctx +
        0.20·cos_pico + 0.10·cos_meth +
        0.20·jac_refs
accept if score ≥ τ (default 0.40)
```

> *Context weight reduced because GraphSAGE dim=256; PICO and Methodology now explicit.*

---

## 6  Tool Registry — Minimal Heavy Calls Only

| Tool                       | Heavy?      | Rationale       |
| -------------------------- | ----------- | --------------- |
| `search_pubmed`            | HTTP API    | network I/O     |
| `embed_text`               | GPU         | Qwen inference  |
| `run_pico`                 | GPU+LLM     | Gemma batch     |
| `prisma_check`             | LLM         | Gemma           |
| `find_existing_sr`         | API+compute | overlap calc    |
| `propose_alternative_pico` | LLM         | heavy reasoning |

Everything else (score calc, add\_papers, recluster, GraphSAGE) is automatic plumbing; Gemma can trigger them indirectly by choosing higher-level verbs (`ensure_completeness`, etc.).

---

## 7  Fallback Logic for “SR Already Exists but Related Gap Possible”

1. `find_existing_sr` → overlap ≥ 0.6 triggers Gemma to call `propose_alternative_pico`.
2. New PICO is embedded; deterministic code launches **a new Retrieval seed** without clearing earlier data (reuse vectors).
3. Old cluster marked “covered\_by\_existing\_SR”, new cluster grows.

---

## 8  Hardware Envelope Check

* 10 M papers × (E: 1024 FP16) = 20 GB mem-map
* Z 256-d ≈ 5 GB if generated, but can be loaded lazily in 1 M-node chunks.
* 32 GB RAM budget suffices if sampling batches ≤ 65 k vectors.

---

## 9  Outstanding Knobs To Freeze

| Parameter                         | Current Default          | Need your OK |
| --------------------------------- | ------------------------ | ------------ |
| token\_soft\_cap                  | 9 500                    |              |
| score τ                           | 0.40                     |              |
| weights (sem/ctx/pico/meth/jac)   | 0.35/0.15/0.20/0.10/0.20 |              |
| min\_eligible\_trials (gap vs SR) | 6                        |              |