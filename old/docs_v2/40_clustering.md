**Filename:** `docs/40_clustering.md`

---

# 40 – Clustering, Cluster Lifecycle & Stability Rules

This memo defines how papers are grouped into “topic kernels”, how those kernels stay stable across expansions, and how Gemma is empowered—but constrained—to restructure them.

---

## 1  Why Clustering Exists

* Give Gemma a **coarse map** so it reasons about hundreds of groups, not millions of papers.
* Supply metrics (size, citation growth, heterogeneity) that drive high-level decisions.
* Act as containers for PICO extraction, PRISMA checks, and eventual SR drafts.

---

## 2  Initial Cluster Formation

### 2.1  Feature Matrix

* Each paper → concatenated vector `[I ‖ C ‖ O ‖ StudyDesign]`.
* Dim = 1024 × 4 + 1024 = 5 120 floats (20 480 B fp16).

### 2.2  HDBSCAN Parameters

| Parameter                  | Value  | Reason                 |
| -------------------------- | ------ | ---------------------- |
| `min_samples`              | 8      | Reject singleton noise |
| `min_cluster_size`         | 30     | Avoid micro-clusters   |
| `metric`                   | cosine | normalised vectors     |
| `cluster_selection_method` | 'eom'  | stable flat hierarchy  |

*Distance matrix computed in 128 K batch stripes to stay below 4 GB RAM.*

Result: list of clusters `C₀ … Cₙ`, plus label `-1` for noise.

---

## 3  Cluster Acceptance & Freezing

1. Output table sorted by descending size.
2. **Gemma “Cluster Audit #1”** prompt contains for each cluster:

   * sample abstracts (token-aware, §30\_embeddings),
   * size, mean year, citation velocity growth rate (CVGR), evidence refresh score (ERS), heterogeneity numeric, etc.
3. Gemma responds JSON:

```json
[
 {"idx":0,"action":"keep","label":"KD vs AED paediatric"},
 {"idx":1,"action":"merge_into", "target_idx":0},
 {"idx":2,"action":"discard"}
]
```

4. Executor applies instructions.  For each **keep**:

```
centroid_E = mean(E_vectors)
centroid_PICO = mean(I‖C‖O‖M)
cluster_id = sha1(label + unix_ts)[:12]
```

Stored in table `cluster_manifest`:

| field             | type      | note                                         |
| ----------------- | --------- | -------------------------------------------- |
| `cluster_id`      | TEXT PK   | frozen                                       |
| `label`           | TEXT      | 10-word Gemma string                         |
| `summary`         | TEXT      | 50-word Gemma string                         |
| `frozen_centroid` | BLOB fp16 | 5 120-dim                                    |
| `tau_assign`      | REAL      | initially 95th percentile of inner distances |
| `parent_id`       | TEXT      | for split lineage                            |
| `created_ts`      | INT       | unix epoch                                   |

---

## 4  Incremental Membership Assignment

For every **new paper** added during expansions:

```python
dist = 1 - cosine(new_vec, frozen_centroid)
if dist <= tau_assign:   assign_to_cluster(cluster_id)
else:                    unassigned_pool.add(pmid)
```

Cost: O(#frozen\_clusters) ≈ few hundred per paper.

---

## 5  Dispersion Monitoring & Split Logic

### 5.1  Dispersion Metric

```
dispersion = 1 - mean_cosine(member_vecs , frozen_centroid)
```

### 5.2  Trigger

*If `dispersion ≥ 0.70` and cluster size ≥ 2× min\_cluster\_size (60), executor tags cluster “unstable”.*
Next Gemma cycle receives list of unstable clusters plus sample abstracts across the spread.

### 5.3  Gemma Split Command

```json
{"cluster_id":"67e9c14b62d3","action":"split_cluster"}
```

Executor re-runs HDBSCAN **inside that cluster’s members** with `min_cluster_size=20`.
Children inherit `parent_id`, get new SHA-IDs and frozen centroids.

---

## 6  Merge Logic

If Gemma decides two clusters semantically overlap:

```json
{"action":"merge_clusters",
 "source":"fa12cd88991e",
 "target":"67e9c14b62d3"}
```

Executor:

1. Concatenates member lists → new centroid = mean of union.
2. Recomputes `tau_assign` as P95 of new distances.
3. Updates size, metrics; marks `source` as `merged_into`.

---

## 7  Per-Cluster Metrics (stored each cycle)

| Metric           | Formula                                              |
| ---------------- | ---------------------------------------------------- |
| **Size**         | count(papers)                                        |
| **CVGR**         | slope of citations/time (least-squares last 3 years) |
| **ERS**          | (# papers published last 18 mo) / size               |
| **het\_numeric** | mean cosine variety of P, I, C, O                    |
| **dispersion**   | as above                                             |
| **sr\_overlap**  | output of `find_existing_sr` (0-1)                   |

These numbers feed Gemma’s triage prompt.

---

## 8  Special Cases

* **Noise cluster (-1)** – ignored unless Gemma explicitly queries “show\_noise”.
* **Very small cluster (< 30)** – Gemma can call `drop_cluster`; papers then remain unassigned until some other cluster accepts them.

---

## 9  Tests

```python
def test_split_reduces_dispersion():
    old_disp = manifest["dispersion"]
    split_cluster(cid)
    new_disp = max(manifest_child["dispersion"] for child in children)
    assert new_disp < old_disp * 0.8
```

---

The clustering subsystem is now fully specified: deterministic rule set, Gemma interaction surface, storage, and monitoring.  Next memo `50_expansion.md` describes how new papers arrive for these clusters.
