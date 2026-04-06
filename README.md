# The Gravity Well: Return-Cost Asymmetry and Hubness in Mental Health Recommender Systems

This repository contains the implementation of the **Gravity Well** audit pipeline, introduced in our paper at the Digital Minds Workshop @ ICWSM 2026.

We propose **Return-Cost Asymmetry (ΔR)** — a model-agnostic audit metric that quantifies asymmetric topological trapping in embedding-based recommender systems. Crisis-related content (depression, suicidal ideation, self-harm) clusters near the spatial mean of the embedding space due to the hubness phenomenon, making it algorithmically easy to drift into and structurally difficult to escape from.

---

## Key Findings

- Under the contrastive encoder (BGE), **98.6% of simulated outward walks** (escaping crisis content) never escape within 200 steps, versus 21.5% of inward walks that fail to arrive — a censoring ratio of **4.59×**
- The general-purpose encoder (MiniLM) shows a milder but significant asymmetry (**1.54×**, step ΔR = +42.28, p < 0.001)
- ΔR **increases** under uniform sampling and persists on a symmetric mutual k-NN graph, confirming the asymmetry is geometric rather than an artifact of the transition kernel or graph directedness
- CSLS hubness mitigation does not dissolve the trap, implicating intrinsic cluster geometry rather than hub dominance alone

---

## Repository Structure

```
gravity-well/
│
├── gravity_well.ipynb          # Main Google Colab notebook (all sections)
├── gravity_well.py             # Python export of the notebook
│
├── data/                       # Created at runtime, saved to Google Drive
│   ├── raw_posts.parquet
│   └── clean_posts.parquet
│
├── embeddings/                 # Created at runtime
│   ├── minilm_vectors.npy
│   └── bge_vectors.npy
│
├── graphs/                     # Created at runtime
│   ├── minilm.index
│   ├── bge.index
│   ├── minilm_knn.npy
│   └── bge_knn.npy
│
├── results/                    # Created at runtime
│   ├── table1_hubness_*.csv
│   ├── table2_trajectories_*.csv
│   ├── table3_extended_*.csv
│   ├── latex_tables_*.tex
│   └── audit_log_*.json
│
└── figures/                    # Created at runtime
    ├── fig1_hubness_*.pdf
    ├── fig2_walk_costs_*.pdf
    ├── fig3_umap_*.pdf
    └── fig4_delta_r_comparison.pdf
```

---

## Pipeline Overview

The notebook is organized into numbered sections. Each expensive section saves a checkpoint sentinel to Google Drive and skips recomputation on re-run.

| Section | Description |
|---------|-------------|
| 0 | Installs |
| 1 | Imports |
| 2 | Config (all hyperparameters) |
| 3 | Google Drive mount and checkpoint system |
| 4 | Data collection from HuggingFace |
| 5 | Preprocessing and anonymization |
| 6 | Model loader and embedding encoding |
| 7 | k-NN graph construction (FAISS) |
| 7B | CSLS distance transformation |
| 7C | Mutual k-NN graph construction |
| 8 | Hubness score computation (H1) |
| 9 | Baseline random walk simulation |
| 9B | Censoring rates and entropy diagnostic |
| 9C | CSLS walk simulation |
| 9D | Uniform sampling ablation |
| 9E | Mutual k-NN walk simulation |
| 10 | ΔR computation and statistical tests |
| 11 | Visualizations |
| 12 | Results export (CSV, LaTeX, audit log) |
| 13 | Negative controls and severity gradient |

---

## Datasets

All datasets are publicly available via HuggingFace:

| Corpus | Source | Label |
|--------|--------|-------|
| Crisis (n=15,000) | `solomonk/reddit_mental_health_posts` | r/depression |
| Crisis supplement | `thePixel42/depression-detection` | label=1 (r/SuicideWatch, r/depression) |
| Normative (n=15,000) | `SocialGrep/one-million-reddit-questions` | r/AskReddit |
| Borderline (n=10,000) | `solomonk/reddit_mental_health_posts` | r/ADHD, r/PTSD, r/OCD |

No individual posts are reproduced in this repository. All text is anonymized (usernames, subreddit names, URLs, emails, and phone numbers replaced with tokens).

---

## Embedding Models

| Key | Model | Dimensions | Description |
|-----|-------|-----------|-------------|
| `minilm` | `all-MiniLM-L6-v2` | 384 | General-purpose sentence transformer |
| `bge` | `BAAI/bge-small-en` | 384 | Contrastive-trained retrieval encoder |

---

## Hyperparameters

All hyperparameters are centralized in `CONFIG` in Section 2.

**Full run (paper results):**
```python
CONFIG = {
    "n_crisis":        15000,
    "n_normative":     15000,
    "n_borderline":    10000,
    "k_neighbors":     20,
    "walk_length":     200,
    "n_walks_inward":  10000,
    "n_walks_outward": 10000,
    "bootstrap_iters": 1000,
    "softmax_temp":    "auto",
    "seed":            42,
}
```

**Smoke test (quick verification):**
```python
CONFIG = {
    "n_crisis":        800,
    "n_normative":     800,
    "n_borderline":    400,
    "n_walks_inward":  1000,
    "n_walks_outward": 1000,
    "bootstrap_iters": 200,
}
```

---

## How to Run

### Requirements

```bash
pip install sentence-transformers faiss-cpu datasets praw \
            scikit-learn scipy matplotlib seaborn tqdm \
            numpy pandas umap-learn
```

### Google Colab (recommended)

1. Open `gravity_well.ipynb` in Google Colab
2. Mount your Google Drive when prompted (Section 3)
3. Run sections 1–12 for MiniLM
4. Change `ACTIVE_MODEL = "bge"` in Section 6
5. Run sections 6–12 for BGE (do **not** restart the runtime between models — `all_model_results` must persist in memory for the cross-model comparison)

### Switching Models

```python
# In Section 6, change this line:
ACTIVE_MODEL = "minilm"   # or "bge"
```

### Clearing Checkpoints

Run the checkpoint clear cell in Section 4 before any full re-run. This deletes all cached data and forces recomputation from scratch.

---

## Key Metrics

**Equation 1 — k-occurrence count:**

$$N_k(x) = \sum_{y \in \mathcal{N} \setminus \{x\}} \mathbf{1}[x \in \text{NN}_k(y)]$$

**Equation 2 — Hubness score:**

$$H_x = \frac{N_k(x) - \mu_{N_k}}{\sigma_{N_k}}$$

**Equation 3 — Inward trajectory cost:**

$$\text{Cost}(\mathcal{T}_\text{in}) = \sum_{i=1}^{m} \text{dist}(v_{i-1}, v_i) \quad \text{s.t.} \quad v_m \in \mathcal{C}_\text{crisis}$$

**Equation 4 — Outward trajectory cost:**

$$\text{Cost}(\mathcal{T}_\text{out}) = \sum_{j=1}^{p} \text{dist}(v_{j-1}, v_j) \quad \text{s.t.} \quad v_p \in \mathcal{C}_\text{norm}$$

**Equation 5 — Return-Cost Asymmetry (primary audit metric):**

$$\Delta R = \mathbb{E}[\text{Cost}(\mathcal{T}_\text{out})] - \mathbb{E}[\text{Cost}(\mathcal{T}_\text{in})]$$

---

## Statistical Tests

- **Mann-Whitney U** (one-tailed, outward > inward) — primary test for step ΔR
- **Permutation null test** — 1,000 label shuffles, empirical p-value
- **Bootstrap 95% CI** — 1,000 resamples on cost ΔR (completed walks only)

---

## Ablations

| Ablation | Section | Description |
|----------|---------|-------------|
| k sensitivity | Table 1 | k ∈ {10, 20, 50} |
| Walk length | Table 1 | length ∈ {100, 200} |
| CSLS mitigation | 9C / Table 6 | Hubness-reduced distances |
| Uniform sampling | 9D / Table 8 | Equal 1/k neighbor probability |
| Mutual k-NN | 9E / Table 8 | Symmetric reciprocal edges |
| Negative controls | 13 / Table 7 | norm→norm, norm→borderline |

---

## Citation

```bibtex
@inproceedings{gravitywelll2026,
  title     = {The Gravity Well: Return-Cost Asymmetry and Hubness
               in Mental Health Recommender Systems},
  author    = {Anonymous},
  booktitle = {Proceedings of the Digital Minds Workshop
               at the International AAAI Conference on Web and Social Media (ICWSM)},
  year      = {2026},
}
```

*Citation will be updated with author names and full proceedings details upon publication.*

---

## Ethics

All data are publicly available Reddit posts accessed via HuggingFace datasets. No private data were collected, no users were contacted, and no intervention was conducted. Crisis content is handled with care — no individual posts are reproduced anywhere in this repository. Text anonymization removes usernames, subreddit names, URLs, email addresses, and phone numbers.

This tool is an audit metric, not a clinical diagnostic. The findings should not be used as clinical guidance.

We acknowledge a dual-use concern: ΔR could theoretically be used to optimize trapping rather than detect it. We believe transparency about these geometric properties is more likely to motivate mitigation than concealment.

---

## License

Code released under the MIT License. See `LICENSE` for details.
