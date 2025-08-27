**Filename:** `docs/90_config.md`

---

# 90 – Master Configuration Reference (`config.toml`)

All numeric knobs, path pointers, and rate limits live in a single TOML file at project root.
You should *never* hard-code constants in the Python packages—read them from this file once at startup.

Copy the template below as `config.toml` and edit as desired.
Any field omitted falls back to the shown default.

```toml
###############################################################################
# INGEST & API
###############################################################################
[ingest]
# PubMed E-utilities
pubmed_batch          = 200       # PMIDs per efetch request
max_qps               = 3         # API calls per second
retry_attempts        = 3
backoff_factor        = 2         # seconds^attempt

# Crossref
crossref_timeout_s    = 20
crossref_retry        = 3

# Commit frequency to SQLite (rows)
insert_commit_every   = 2000

# Baseline XML snapshot path
baseline_dir          = "baseline_xml/"

###############################################################################
# EMBEDDING
###############################################################################
[embedding]
chunk_size            = 50000     # abstracts per GPU batch
refresh_threshold     = 10000     # new papers before embedding pass
model_name            = "Qwen-0.6B-instruct"
fp16                  = true

# Token budget for Gemma prompt
token_soft_cap        = 9500

###############################################################################
# CLUSTERING
###############################################################################
[clustering]
min_samples           = 8
min_cluster_size      = 30
dispersion_split      = 0.70       # trigger for Gemma split
tau_assign_fallback   = 0.20       # if P95 not computable

###############################################################################
# EXPANSION – SEMANTIC BOUNDARY
###############################################################################
[expansion.semantic]
tau_start             = 0.60
delta_tau             = 0.05
boundary_batch_size   = 40
no_keep_limit         = 0          # stop when Gemma keeps <= this

###############################################################################
# EXPANSION – UPSTREAM / RIPPLE
###############################################################################
[expansion.upstream]
alpha_coverage        = 0.25
max_parents           = 30
gemma_reject_threshold= 0.80       # % rejected to trigger halving alpha
tau_sem_child         = 0.45       # cosine gate for downstream seeds

###############################################################################
# HARDWARE GUARDS
###############################################################################
[hardware]
ram_reserve_bytes     = 2147483648     # 2 GiB
gpu_batch_halving     = true

###############################################################################
# TOOLS
###############################################################################
[tools]
# enable/disable heavy tools for debugging
enable_search_pubmed          = true
enable_run_pico               = true
enable_prisma_check           = true
enable_find_existing_sr       = true
enable_propose_alternative_pico = true

###############################################################################
# LOGGING & ROTATION
###############################################################################
[logging]
log_dir               = "logs/"
retain_days           = 30
zip_old_logs          = true

###############################################################################
# GOAL EXECUTION
###############################################################################
[goal]
loop_delay_s          = 5
max_tool_calls        = 1000            # hard safety cap per goal

###############################################################################
# SYSTEMATIC REVIEW THRESHOLDS
###############################################################################
[sr]
min_eligible_trials   = 6
prisma_mandatory_items = [4,5,6,7,8,9,10]
```

---

## How to Change a Value

1. Open `config.toml`, edit the field.
2. Restart the agent—modules read config only at launch.
3. The new value is logged in `Cycle 0` header for traceability.

---

## Programmatic Access Example

```python
import tomllib, pathlib
cfg = tomllib.load(open(pathlib.Path("config.toml"), "rb"))
tau0 = cfg["expansion"]["semantic"]["tau_start"]
```

---

### End of master config.

This concludes the architecture documentation set.
You can now proceed to coding or paste code snippets for alignment review.
