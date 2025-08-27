**Filename:** `docs/80_guardrails_logging.md`

---

# 80 – Guard-Rails, Error Resilience & Logging Strategy

The agent must run unattended for days without corrupting state or exhausting resources.
This memo enumerates every fail-safe, retry policy, and log practice that keeps the system self-healing and auditable.

---

## 1  Memory & Token Safety

| Guard               | Trigger                                    | Action                                                                           |
| ------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- |
| **RAM guard**       | `psutil.virtual_memory().available < 2 GB` | Abort expansion loop; mark `"expansion_halt":"RAM"` in log and cluster manifest. |
| **GPU VRAM guard**  | CUDA OOM during embedding                  | Halve batch size and retry once; if still OOM, skip batch and log.               |
| **Gemma token cap** | Prompt > 9 500 tokens                      | Oldest abstracts trimmed until cap met.                                          |

---

## 2  API Reliability

| API              | Retry policy                | Offline fallback                                          |
| ---------------- | --------------------------- | --------------------------------------------------------- |
| NCBI E-utilities | 3 retries, back-off `2^n` s | PubMed Baseline XML snapshot queried via SQLite mirror    |
| Crossref         | 3 retries, 10-s wait        | Skip refs; upstream parent step logs `"crossref_offline"` |

If an API remains unreachable for **>24 h**, executor sets goal\_state `"status":"paused_api"` and sleeps until manual restart.

---

## 3  JSON Robustness

* Gemma responses parsed with `json5` (allows trailing commas).
* On `JSONDecodeError` executor sends *one* repair prompt:

```
System: "Return ONLY valid JSON for the prior message."
```

If second attempt fails ➜ write `"json_error":true` to log; cycle ends without state change.

---

## 4  Log Structure

Logs live in `logs/YYYY-MM-DD_cycleN.md`.

### 4.1 Markdown Template

```
## Cycle 17  (2025-08-04 14:22 UTC)

### Goal
goal_id: 6a4f46a1b2e6   primary: seek_gap   topic: KD epilepsy adults

### Gemma Prompt (truncated)
> ...
### Gemma Response
> Thought: ...
> Action: { ... }

### Executor Notes
- tool=search_pubmed   cand=2000  accepted=123
- expansion_halt: RAM
- json_status: ok

### Resource Stats
RAM used: 13.1 GiB  VRAM used: 4.7 GiB  Δpapers: +123
```

* Chain-of-thought retained verbatim; redact manually before sharing.

### 4.2 Rotation

* At midnight local time, previous day’s log folder gzipped (`2025-08-04.zip`).
* Zip deletion after 30 days—configurable (`log_retention_days`).

---

## 5  STOP & PAUSE Controls

* **Hard STOP:** create empty file `STOP` in project root ➜ executor finishes current cycle, sets goal\_state.status=`"stopped_manual"`, exits.
* **Pause until resume:** create file `PAUSE` ➜ executor saves state and sleeps; remove file to resume.

No interactive shell required; safe for headless screen sessions.

---

## 6  History & Audit Trail

Every change to goal\_state.json appended to `history`:

```json
{"cycle":21,"event":"merge_clusters",
 "source":"fa12cd8...","target":"67e9c14..."}
```

History persists across restarts (SQLite WAL ensures atomic write).

---

## 7  Crash Recovery

* Executor catches top-level exceptions, logs traceback, sets goal\_state `"status":"crashed"`, exits.
* On next launch, if status=`"crashed"` or `"paused_api"` ➜ auto-resume cycle counter+1.
* In-progress embedding batches are idempotent: vector append writes go to temp file then `os.rename()`.

---

## 8  Security Posture (local environment)

* No external inbound ports opened.
* API keys stored in `.env`; loaded via `python-dotenv`.
* Raw API JSON cached locally; remove before public release.

---

The guard-rail framework now makes the core loop self-recovering, memory-bounded, and fully auditable.  Proceed to `90_config.md` for the canonical list of all tunable knobs.
