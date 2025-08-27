**Filename:** `docs/70_tools.md`

---

# 70 – Heavy-Tool Registry & Execution Contracts

Gemma can accomplish nothing in the external world unless it calls **tools**.
Each tool encapsulates an expensive or network-bound operation implemented by deterministic code.
Gemma’s access is restricted to this registry; any unregistered or malformed call is rejected.

---

## 1 Common Call Syntax

Gemma must terminate its response with a single JSON block:

```json
Action: {
  "tool": "<name from registry>",
  "args": { ... }
}
```

* Only top-level keys `"tool"` and `"args"` accepted.
* `args` must match schema exactly; missing keys use defaults where defined.
* Executor streams tool output back to Gemma on the next cycle if relevant.

---

## 2 Tool Catalogue

### 2.1 `search_pubmed`

| Aspect      | Details                                                                           |              |
| ----------- | --------------------------------------------------------------------------------- | ------------ |
| Purpose     | Run instruction-style query against PubMed; return candidate abstracts.           |              |
| Args schema | \`{ "query": str, "retmax": int=2000, "expand\_mode": "semantic"                  | "ripple" }\` |
| Output      | List of `(pmid, title, abstract)` triples saved to temp table `candidate_papers`. |              |
| Cost        | 1 HTTP roundtrip + embedding batch.                                               |              |
| Notes       | Uses NCBI *esearch* + *efetch*; rate-limited to 3 QPS.                            |              |

### 2.2 `embed_text`

| Aspect      | Details                                                                                   |                |
| ----------- | ----------------------------------------------------------------------------------------- | -------------- |
| Purpose     | Produce 1024-d Qwen vector(s) for arbitrary text—used when Gemma wants cosine on the fly. |                |
| Args schema | \`{ "text": str                                                                           | list\[str] }\` |
| Output      | Numpy array saved to RAM cache for same cycle.                                            |                |
| Cost        | ≤ 10 ms per string; but triggers GPU load.                                                |                |
| Limits      | Max 256 strings per call; else executor splits silently.                                  |                |

### 2.3 `run_pico`

\| Purpose | Extract P, I, C, O, StudyDesign fields for a cluster or list of PMIDs using Gemma batch prompt. |
\| Args | `{ "cluster_id": str }` **or** `{ "pmid_list": list[int] }` |
\| Output | JSONL file `pico/<cluster_id>.jsonl` where each line `{"pmid":…, "P":…, ...}` |
\| Cost | \~1.5 s per 40 abstracts (10 k token prompt). |
\| Guard | Executor asks confirmation if batch > 800 abstracts. |

### 2.4 `prisma_check`

\| Purpose | Qualitative compliance check for PRISMA items 4–8. |
\| Args | `{ "cluster_id": str }` |
\| Output | `{"compliant": bool, "missing_items": [4,7]}` |
\| Cost | Single Gemma call (\~300 tokens). |

### 2.5 `find_existing_sr`

\| Purpose | Detect overlap with prior systematic reviews. |
\| Args | `{ "cluster_id": str }` |
\| Implementation | 1) Build “systematic\[sb]” PubMed query from cluster label. 2) Embed titles of returns. 3) cosine against cluster centroid. 4) Return fraction ≥ 0.75 cosine. |
\| Output | `{"overlap": float, "sr_pmids": [ ... ]}` |

### 2.6 `propose_alternative_pico`

\| Purpose | When overlap high, ask Gemma to craft new PICO theme from leftover papers. |
\| Args | `{ "exclude_pmids": list[int] }` |
\| Output | `{"new_PICO": str}` |
\| Follow-up | Executor spins a **new handbook** with `Seed_query` = `new_PICO`. |

---

## 3 Light Helpers (not exposed)

`add_papers`, `compute_metrics`, `recluster`, `assign_cluster` run automatically; Gemma never references them.

---

## 4 Error Handling

| Error                | Executor Response                                                                     |
| -------------------- | ------------------------------------------------------------------------------------- |
| Unknown tool         | Write `{"error":"unknown_tool"}` to log; cycle ends with no state change.             |
| Missing arg key      | Same, plus status `error` in goal\_state.                                             |
| API 502 / timeout    | Retry 3× with back-off; on final fail return `"api_offline":true` so Gemma can react. |
| GPU OOM during embed | Halve batch, retry; second failure → log & skip batch.                                |

---

## 5 Security Note

Tools never accept raw file paths from Gemma; executor controls disk writes.
External URLs are limited to approved domains (`ncbi.nlm.nih.gov`, `api.crossref.org`).
This prevents prompt-injection file writes or SSRF.

---

## 6 Adding a New Tool (developer checklist)

1. Implement deterministic Python function, ensure idempotence.
2. Append entry to this memo with schema.
3. Add schema validation in `tool_executor.py`.
4. Update `90_config.md` if new resource limits required.
5. Mention the tool in future handbooks—Gemma learns it automatically from prompt.

---

Tools are now formalised; Gemma‐executor interface is stable.  Proceed to `80_guardrails_logging.md` for operational safety nets.
