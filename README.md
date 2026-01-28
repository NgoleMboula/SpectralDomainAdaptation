# Beyond Mapping : Domain-Invariant Representations via Spectral Embedding of Optimal Transport Plans

This repository contains the official implementation of the paper **"Beyond Mapping : Domain-Invariant Representations via Spectral Embedding of Optimal Transport Plans"**, authored by Abdel Djalil Sad Saoud, Fred Maurice Ngolè Mboula, and Hanane Slimani. Accepted at the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) [[paper]](https://arxiv.org/pdf/2601.13350).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Run](#run)
- [Citation](#citation)
- [References](#references)

## Overview

This paper presents a novel Optimal Transport-based domain adaptation method that interprets optimal transport plans as adjacency matrices capturing cross-domain connectivity, then derives domain-invariant and discriminative representations of samples via spectral embedding [[1]](#1).

**Key contributions:**

1. **Novel OT-based framework:** We propose a domain adaptation approach that leverages the cross-domain connectivity captured by the transport plans to compute domain-invariant and discriminative representation of samples, rather than estimating a mapping from one domain to another in the samples space.

2. **Multi-source extension:** We extend our framework to multi-source domain adaptation scenarios, inspired from [[2]](#2).

3. **Comprehensive evaluation:** We evaluate our method on acoustic adaptation benchmarks and demonstrate industrial relevance through a cable defect diagnosis use case based on Time Domain Reflectometry.

## Installation

**Development Environment:**
- Python 3.10.0
- pip 25.3

**Install all required packages:**
```bash
pip install -r requirements.txt
```

## Run

To run SeOT

### Step 1. Prepare Data
Download datasets to the `data/` directory.

### Step 2. Run Experiments
**For MSD/MGR benchmarks:**
```bash
# On MSD
python main_seot_msda.py --benchmark MSD --algorithm SeOT --reg_e 1e-4 --reg_e_bar 1e-2 --n_component 10 --epochs 100 --lr 0.001 --batch-size 128

# On MGR
python main_seot_msda.py --benchmark MGR --algorithm SeOT --reg_e 1e-4 --reg_e_bar 1e-2 --n_component 10 --epochs 100 --lr 0.001 --batch-size 128
```
**For CS-RT benchmark:**
```bash
python main_seot_reflecto.py --algorithm SeOT

```

## Citation

If you use this code or find our work useful, please consider citing our paper:

```bibtex
@misc{saoud2026mappingdomaininvariantrepresentations,
      title={Beyond Mapping : Domain-Invariant Representations via Spectral Embedding of Optimal Transport Plans}, 
      author={Abdel Djalil Sad Saoud and Fred Maurice Ngolè Mboula and Hanane Slimani},
      year={2026},
      eprint={2601.13350},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.13350}, 
}
```

## References

<a id="1">[1]</a> scikit-learn developers. (2023). sklearn.manifold.SpectralEmbedding. In *scikit-learn Documentation*. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html.

<a id="2">[2]</a> Montesuma, E. F. & Mboula, F. M. N. (2021). Wasserstein Barycenter Transport for Domain Adaptation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 15794-15803).