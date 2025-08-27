**Filename:** `docs/20_data_ingest.md`

---

# 20 – Data Ingest, Normalisation & Storage

*(building a reproducible PubMed + Crossref corpus on local disk)*

---

## 1  High-Level Flow

```
PubMed E-utilities       Crossref REST
        │                       │
        ▼                       ▼
   fetch_medline()        fetch_crossref()
        │                       │
        ├─→ parse_article() ─┐   │
        │                   │   │
        └─→ insert_sqlite() │   │
                            └───┴──► dedup_merge_refs()
                                            │
                                            ▼
                                 mmap embeddings folder
```

Everything runs from a single CLI command:

```bash
python -m ingest --since 2023-01-01 --retmax 100000
```

---

## 2  SQLite Schema v1

### 2.1 `paper` table

| column           | type    | note                                     |
| ---------------- | ------- | ---------------------------------------- |
| pmid PRIMARY KEY | INTEGER | NCBI PubMed ID (if absent, use DOI hash) |
| doi              | TEXT    | normalised, lowercase                    |
| title            | TEXT    | UTF-8                                    |
| abstract         | TEXT    | UTF-8                                    |
| journal          | TEXT    | NLM journal title abbreviation           |
| year             | INT     | publication year                         |
| article\_type    | TEXT    | “Randomized Controlled Trial”, “Review”… |
| created\_ts      | INT     | Unix epoch seconds (first seen)          |

### 2.2 `ref` table

| column               | type    | meaning                                      |
| -------------------- | ------- | -------------------------------------------- |
| src                  | INTEGER | citing PMID                                  |
| dst                  | INTEGER | cited PMID (if DOI-only, map to PMID or 0)   |
| prov                 | INT     | bitmask 1 = PubMed ‘cites’, 2 = Crossref ref |
| PRIMARY KEY(src,dst) |         | ensures dedup                                |

---

## 3  PubMed Fetcher

```python
def fetch_medline(pmids: list[int]) -> list[ET.Element]:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db":"pubmed","id":','.join(map(str,pmids)),
              "retmode":"xml","rettype":"abstract"}
    xml = requests.get(url, params=params, timeout=20).text
    return parse_etree(xml)
```

* **Batch size** 200 IDs
* **Retry** 3 attempts with 2× back-off
* **Rate** ≤ 3 QPS default (configurable)

`parse_article()` extracts title, abstract, article type, DOI, refs (via `<ReferenceList>`).

---

## 4  Crossref Reference Harvester

```python
def fetch_crossref(doi: str) -> list[str]:
    url = f"https://api.crossref.org/works/{doi}"
    j = requests.get(url, timeout=20).json()
    return [ref["DOI"].lower()
            for ref in j["message"].get("reference", [])
            if "DOI" in ref]
```

* Called **lazily** the first time Gemma asks for upstream parents of a paper lacking reference data.
* Cached to `cache/crossref_json/{sha1(doi)}.json`.

Mapping DOI→PMID uses NCBI `elink` (`dbfrom=pubmed&linkname=pubmed_pubmed_doi`); unmapped DOIs stored with `dst=0`.

---

## 5  Dedup & Merge Logic

```sql
-- insert paper row
INSERT OR IGNORE INTO paper VALUES (...)

-- insert each ref edge
INSERT OR IGNORE INTO ref(src,dst,prov) VALUES (?,?,?)
```

* If an edge already exists with different provenance, `prov = prov|new`.
* Chronology inconsistencies (src.year < dst.year) are kept; clustering algorithm does not depend on DAG.

---

## 6  Embedding Pipeline Trigger

After every **N = 10 000** new `paper` rows:

1. Query uncached PMIDs → list\[abstract]
2. Run Qwen encoder on GPU in chunks of 50 k texts
3. Append vectors to `emb/E.npy` using `numpy.memmap`
4. Store row-offset mapping in `emb/offsets.npy` (`pmid → index`)

Extraction vectors (`p.npy` … `m.npy`) appear only **after** Gemma’s first `run_pico` batch.

---

## 7  Baseline Snapshot & Offline Mode

* Weekly cron job downloads NIH **PubMed Baseline** (≈ 210 XML gzip files).
* A small script `baseline_to_sqlite.py` pre-populates `paper` and `ref` tables—used if live APIs fail 5× consecutively.

Runtime switch:

```python
if api_error_count > 5:
    use_baseline = True
```

---

## 8  Configuration Options (`config.toml` Extract)

```toml
[ingest]
pubmed_batch        = 200
max_qps             = 3
crossref_timeout_s  = 20
insert_commit_every = 2000

[embedding]
chunk_size          = 50000
refresh_threshold   = 10000     # new papers before re-embed
```

---

## 9  Tests

| test                           | pass criteria                                |
| ------------------------------ | -------------------------------------------- |
| `test_fetch_medline_roundtrip` | 95% abstracts non-empty for 500 PMIDs        |
| `test_doi_mapping`             | ≥ 85 % of incoming Crossref DOIs map to PMID |
| `test_ref_dedup`               | duplicate insert does not increase rowcount  |
| `test_embed_append`            | file size grows by 1024 × len(batch) × 2 B   |

---

### The ingest layer now feeds consistent `paper` and `ref` tables and a growing `E.npy`.

All later modules (clustering, expansion, Gemma processing) depend only on these artefacts.
