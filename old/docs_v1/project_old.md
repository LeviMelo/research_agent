This is the final, comprehensive plan for the **Autonomous Research Agent (ARA)**.

---

### **Project Goal: The Autonomous Discovery of Meta-Analysis Gaps**

To build a persistent, self-directing software agent that autonomously explores the scientific citation network to identify, validate, and propose novel, high-potential opportunities for systematic reviews and meta-analyses.

### **Core Architecture: The Agent-Centric Framework**

The system is not a linear pipeline but a stateful agent operating in a continuous Sense-Think-Act loop. It runs in sessions on a local machine, saves its state, and becomes progressively smarter over time.

*   **World Model:** A persistent **Knowledge Graph (`G`)** stored on disk, containing all known papers, citations, and their calculated attributes. This is the agent's long-term memory.
*   **Sensory Cortex:** A **Metric Engine** that runs computationally cheap analyses on the graph to detect signals of change and opportunity.
*   **Strategic Core:** A **Large Language Model (Gemma 3n)** acting as the central reasoning engine, making strategic decisions based on sensory input.
*   **Action Toolkit:** A library of "expensive" functions the agent can choose to execute, such as deep PICO analysis or graph expansion.
*   **Logging System:** An immutable log of all LLM prompts and responses for auditing and multi-model validation.

---

### **Phase 1: Knowledge Graph Construction & Representation (The World)**

**Objective:** To build and maintain a rich, multi-modal representation of the scientific literature.

1.  **Node Definition:** Each paper (`v`) in the graph `G` is a node, uniquely identified by its DOI/PMID.
2.  **Edge Definition:** A directed edge `(u, v)` represents that paper `u` cites paper `v`.
3.  **Multi-Faceted Feature Vector (`X_v`):** Every node is described by a concatenated feature vector containing:
    *   **Semantic Features:** A text embedding of the title/abstract (e.g., from `qwen3-0.6b`).
    *   **Temporal Features:** Normalized publication year and age.
    *   **Impact Features:** Total citation count.
    *   **Type Features:** A binary `is_review` flag derived from PubMed metadata.
4.  **Context-Aware Embedding (`Z_v`):** Using the **GraphSAGE** framework, we generate a final, context-aware embedding for each node. This is the core representational learning step.
    *   **Process:** GraphSAGE combines the node's own feature vector (`X_v`) with features aggregated from its local neighborhood.
    *   **Customization:** We will employ **intelligent neighbor sampling**, selecting neighbors based on a heuristic (e.g., Top-K by citation count) to proactively filter noise from spurious citations.
    *   **Inductive Nature:** A trained GraphSAGE model can generate embeddings for new nodes without full retraining, enabling efficient graph expansion.

### **Phase 2: Topic Kernel Lifecycle Management (The Agent's "Lenses")**

**Objective:** To dynamically identify and manage coherent research topics as they evolve within the graph.

1.  **Topic Kernel Definition:** A Topic Kernel (`K_C`) is an analytical construct, not a fixed object. It is defined as all nodes within a certain embedding distance of a cluster centroid, making it robust to imperfect clustering (HDBSCAN).
2.  **Lifecycle Events:** The agent will periodically run checks to manage these kernels:
    *   **Growth:** Kernels are updated by re-running the radius search after the graph in their vicinity expands.
    *   **Merge:** If two kernel centroids become too similar, they are merged into a single, new kernel representing a field convergence.
    *   **Split (Partition):** If a kernel's internal semantic dispersion becomes too high, it is fractured into smaller, more coherent sub-topic kernels.
    *   **Extinguish:** If a kernel's momentum metrics (like CVGR) die down and its synthesis score is high, it is archived as "obsolete" or "solved."

### **Phase 3: The ARA's Operational Loop (The "Sense-Think-Act" Cycle)**

This is the main execution flow of the agent.

1.  **SENSE (Triage):**
    *   Upon starting a session, the **Metric Engine** runs a suite of computationally cheap metrics on all active Topic Kernels.
    *   The dashboard of metrics includes: **Semantic Dispersion**, **Internal Density**, **Citation Velocity Growth Rate (CVGR)**, **Citation Entropy**, **Synthesis Saturation Score (S3)**, and **Evidence Refresh Score (ERS)**.
    *   These metrics are compiled into a **"Triage Report"** highlighting the most interesting topics.

2.  **THINK (Strategy):**
    *   The Triage Report is passed to the **Gemma 3n Agent Core**.
    *   The agent's prompt asks it to choose a single, strategic action from a predefined list based on the metrics. This is the core decision-making step.
    *   **Action Space:**
        *   `PICO_DEEP_DIVE`: Trigger the final, expensive analysis on a highly promising topic.
        *   `EXPAND_DOWNSTREAM`: Expand the graph from a "hot" topic's child frontier (papers that cite it).
        *   `EXPAND_UPSTREAM`: Expand the graph from a topic's parent frontier to understand its roots.
        *   `VALIDATE_EXTERNALLY`: Formulate and execute a PubMed MeSH query as a sanity check.
        *   `DEPRIORITIZE`: Mark a topic as uninteresting.

3.  **ACT (Execution):**
    *   The system parses the agent's JSON decision and executes the chosen action from its **Action Toolkit**.
    *   If an `EXPAND` action is chosen, new nodes are added to the graph, their features are engineered, and their `Z_v` embeddings are generated inductively. The agent then dynamically adds a new task to its own priority queue to analyze the newly expanded region.

### **Phase 4: The Meta-Analysis Gap Deep Dive (The Final Goal)**

This is the most expensive action, only triggered when the agent has high confidence.

1.  **Granular Comminution:** For a chosen Topic Kernel, Gemma 3n is deployed at scale to perform **per-article PICO extraction**, creating a structured database of the evidence components.
2.  **Generative Question Synthesis:** The agent analyzes this PICO database and **creatively synthesizes a novel, unifying PICO question** that could form the basis of a new meta-analysis, along with a list of compatible primary research papers.
3.  **Final LLM Judgment:** A final dossier is presented to the **Gemma 3n agent as the "Journal Editor"**. This dossier includes:
    *   The machine-generated PICO question.
    *   The state of the evidence, framed using the **Evidence Refresh Score (ERS)** logic ("X% of the evidence is new since the last major review...").
    *   Abstracts of the key papers.
    *   The results of the external MeSH query check.
4.  **The Actionable Output:** The agent makes a final, reasoned **"GREENLIT"** or **"REJECTED"** decision, providing a justification. This constitutes a fully validated, machine-discovered opportunity for a new meta-analysis, ready for human review.

This comprehensive plan establishes a dynamic, persistent, and intelligent system that leverages graph learning for structure, cheap metrics for triage, and a powerful LLM for strategic decision-making and final, nuanced judgment, directly realizing the sophisticated vision we have developed.