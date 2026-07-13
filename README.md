# Investigating the Double Descent Phenomenon

### Factorization Models on Sparse E-Commerce Transaction Data

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Domain](https://img.shields.io/badge/Domain-Collaborative%20Filtering-8A2BE2)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Course project for **Statistical Machine Learning**, M.Sc. in Data Science,
Sapienza University of Rome — Academic Year 2025–2026.

We ask whether the **rank-wise double descent** that Kawakami & Sugiyama (RecSys 2021)
demonstrated for over-parameterized Non-Negative Matrix Factorization on *explicit*
ratings survives the passage to **sparse, one-class, implicit-feedback** data, where
performance must be judged by ranking rather than reconstruction. We sweep the latent
dimension `k` from the under- to the heavily over-parameterized regime across three
matrix-factorization models and five user-activity filters, on real Shopify transaction
data from the Chilean branch of *Fondo de Cultura Económica*.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Findings](#key-findings)
- [The Three Models](#the-three-models)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Notebooks](#running-the-notebooks)
- [Report and Slides](#report-and-slides)
- [References](#references)
- [Authors](#authors)
- [License](#license)

---

## Motivation

In the classical bias–variance picture, test error is U-shaped in model complexity and
large models are expected to overfit. Modern over-parameterized models instead often
*interpolate* the training data and keep generalizing: test error rises near the
interpolation threshold and then descends a second time — the **double descent** curve.

Matrix factorization is a natural laboratory for this question: the latent dimension `k`
controls capacity directly, so a single sweep over `k` traces the model from the
under- to the over-parameterized regime. The methodological goal of the project is not
to build the best recommender, but to isolate **which modeling ingredient the phenomenon
actually depends on** when the feedback is implicit and one-class.

## Key Findings

- **Double descent survives, but conditionally.** It carries over from explicit ratings /
  reconstruction error to sparse, one-class, implicit feedback and ranking evaluation —
  provided a few legible conditions hold.
- **The critical peak is located by parameter counting.** With `k(n+m)` parameters against
  `|Ω|` constraints, the test peak sits near `k★ ≈ |Ω| / (n + m)`, and the observed peak
  sits at or below `k★`, approaching it as per-user density grows.
- **The treatment of negatives is decisive.** The two *sampled-support* models (NMF and
  Logistic MF) show double descent at every filter; the *full-matrix* model (iALS) does not
  at low activity, with a clean, sustained recovery appearing only for denser populations
  (`τ ≥ 20`). What matters is the existence of a fixed finite training support, not the
  loss family.
- **The signal-carrying metric depends on the loss.** For NMF and iALS the ranking metric
  (NDCG) is the more expressive one; for Logistic MF, whose loss is calibrated, the
  pointwise metric (RMSE) traces the curve most cleanly.
- **Regularization modulates amplitude, not existence.** Weakening the NMF penalty from
  `λ = 0.06` to `λ = 0.01` raises the RMSE peak (`≈ 0.47 → ≈ 0.82`) without moving its
  location and without changing the NDCG curve.
- **The low-rank restriction is too conservative here.** Under NMF, the best recommendations
  on every metric come from the *largest* model tested.

## The Three Models

The three formulations differ precisely in how they treat unobserved cells and in the loss
they minimize, which is exactly what lets us attribute the phenomenon to one ingredient.

|                     | NMF                | Weighted MF / iALS      | Logistic MF          |
| ------------------- | ------------------ | ----------------------- | -------------------- |
| **Training support**| Sampled `Ω`        | Full `n × m` matrix     | Sampled `Ω`          |
| **Loss**            | Squared error      | Weighted squared error  | Binary cross-entropy |
| **Non-negativity**  | Yes                | No                      | No                   |
| **Main DD evidence**| RMSE + NDCG        | NDCG, high `τ`          | RMSE                 |
| **Main sensitivity**| `τ`, `λ`           | `τ`, iterations         | `τ`, optimization    |

- **NMF** — reconstructs the binary matrix on a fixed sampled support `Ω = P ∪ N`
  (positives plus `r` sampled negatives per positive), trained with masked multiplicative
  updates under Tikhonov regularization.
- **Weighted MF / iALS** — the canonical implicit-feedback model (Hu, Koren & Volinsky,
  2008); fits every cell of the matrix, distinguishing positives from unobserved cells
  through a per-cell confidence weight `Cui = 1 + αXui` rather than through sampling.
  Implemented with the [`implicit`](https://github.com/benfred/implicit) library.
- **Logistic MF** — abandons reconstruction and models the *probability* of interaction
  with a logistic loss and negative sampling.

## Dataset

Transaction data from the Shopify online store of *Fondo de Cultura Económica Chile*,
covering **October 2019 – December 2025**. Because the online store is only one of several
sales channels, an unobserved online interaction is treated as a **missing preference**,
not a true negative.

| Quantity                         | Value    |
| -------------------------------- | -------- |
| Users                            | 23,605   |
| Items                            | 4,405    |
| Positive user–item interactions  | 136,832  |
| Matrix density                   | 0.13%    |
| Matrix sparsity                  | 99.86%   |

The binary utility matrix is `Xui = 1` if user `u` purchased item `i`, else `0`. Activity
filters `τ ∈ {3, 5, 10, 20, 25}` (minimum interactions per user) produce progressively
denser sub-populations.

> **Note on data availability.** The raw customer transaction data is **not** included in
> this repository for privacy reasons. `notebooks/01_build_interactions.ipynb` documents
> the full pipeline that turns the raw Shopify exports into the modeling file
> `interactions_real_long.csv`. See [`data/README.md`](data/README.md) for the expected
> schema.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_build_interactions.ipynb   # Raw Shopify exports → interactions_real_long.csv
│   ├── 02_nmf.ipynb                  # Non-Negative Matrix Factorization (Colab)
│   ├── 03_weighted_mf_ials.ipynb     # Weighted MF / implicit ALS (Colab)
│   └── 04_logistic_mf.ipynb          # Logistic Matrix Factorization (Kaggle GPU)
├── report/
│   └── SML_Report_DoubleDescent.pdf  # Full written report
├── slides/
│   └── SML_Press_DoubleDescent.pdf   # Presentation slides
├── data/
│   └── README.md                     # Data schema and availability
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

The project was developed across three environments (local, Google Colab, Kaggle). To run
locally:

```bash
git clone https://github.com/<your-username>/double-descent-collaborative-filtering.git
cd double-descent-collaborative-filtering

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running the Notebooks

The notebooks are meant to be run **in order**. They target different environments, noted
below.

| Notebook                        | Environment      | Notes |
| ------------------------------- | ---------------- | ----- |
| `01_build_interactions.ipynb`   | Local            | Reads raw `Sales_until_*.xlsx` / `Product.xlsx` exports and writes `interactions_real_long.csv`. |
| `02_nmf.ipynb`                  | Google Colab     | Uses `drive.mount('/content/drive')`; adjust the data path to your Drive. |
| `03_weighted_mf_ials.ipynb`     | Google Colab     | Same as above; uses the `implicit` library on CPU. |
| `04_logistic_mf.ipynb`          | Kaggle (GPU)     | Requires a CUDA accelerator; reads the dataset from `/kaggle/input`. |

**Adjusting the data path in notebook 01.** The pipeline notebook currently points to a
local absolute directory. Before running it, set `PROJECT_DIR` (top of the second code
cell) to your own location, e.g. a relative path inside the repo:

```python
from pathlib import Path

PROJECT_DIR = Path("data")            # instead of the hardcoded absolute path
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data_final"
OUTPUT_FILE = PROCESSED_DATA_DIR / "interactions_real_long.csv"
```

For the Colab and Kaggle notebooks, update the data-loading path to wherever you place
`interactions_real_long.csv` (a mounted Drive folder or an attached Kaggle dataset).

## Report and Slides

- 📄 **Report:** [`report/SML_Report_DoubleDescent.pdf`](report/SML_Report_DoubleDescent.pdf)
- 🖥️ **Slides:** [`slides/SML_Press_DoubleDescent.pdf`](slides/SML_Press_DoubleDescent.pdf)

## References

1. Y. Hu, Y. Koren, C. Volinsky. *Collaborative Filtering for Implicit Feedback Datasets.* ICDM, 2008.
2. C. C. Johnson. *Logistic Matrix Factorization for Implicit Feedback Data.* NeurIPS Workshop on Distributed ML and Matrix Computations, 2014.
3. Y. Kawakami, M. Sugiyama. *Investigating Overparameterization for Non-Negative Matrix Factorization in Collaborative Filtering.* RecSys, 2021.
4. P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever. *Deep Double Descent: Where Bigger Models and More Data Hurt.* ICLR, 2020. arXiv:1912.02292.
5. N. Gillis. *The Why and How of Nonnegative Matrix Factorization.* 2014. arXiv:1401.5226.

## Authors

Group project for Statistical Machine Learning, Sapienza University of Rome (2025–2026):

- Gianluca Pasquinelli
- Matías Caleb Cerda Barrera
- Denise Brancaleone
- Marco Restuccia

## License

Released under the [MIT License](LICENSE). The raw transaction data is not covered by this
license and is not distributed.
