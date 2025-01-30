<!-- markdownlint-disable first-line-h1 -->
<div align="center">
  <img src="images/logo.png" width="60%" alt="AutoGraph" />
</div>

# AutoGraph: Transformers are Scalable Graph Generators

This repository implements AutoGraph presented in the following paper:

>Dexiong Chen, Markus Krimmel, and Karsten Borgwardt.
[Flatten Graphs as Sequences: Transformers are Scalable Graph Generators][1], Preprint 2025.

**TL;DR**: A novel autoregressive framework that uses decoder-only transformers to generate large attributed graphs.

## Overview

At the core of AutoGraph is a reversible "flattening" process that transforms graphs into random sequences. By sampling and learning from these sequences, AutoGraph enables transformers or any language models to model and generate complex graph structures in a manner akin to natural language. The sequence sampling complexity and length scale linearly with the number of edges, making AutoGraph highly scalable for generating large sparse graphs. Empirically, it achieves state-of-the-art performance across diverse synthetic and molecular graph generation benchmarks, while delivering a 100-fold generation and a 3-fold training speedup compared to leading diffusion models. Furthermore, it demonstrates promising transfer capabilities and supports substructure-conditioned generation without additional fine-tuning. 

The flattening process rely on sampling a sequence of random trail segments with neighborhood information (i.e. a SENT), by traversing the graph through a strategy similar to depth-first search. More details can be found in Algorithm 1 in our [paper][1]. The obtained sequence is then tokenized into a sequence of tokens which can be modeled effectively with a transformer.

<p align="center">
  <img width="80%" src="images/overview.png">
</p>

## Installation

We recommend the users to manage dependencies using [miniconda](https://docs.conda.io/projects/miniconda/en/latest) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
# Replace micromamba with conda if you use conda or miniconda
micromamba env create -f environment.yaml 
micromamba activate autograph
pip install -e .
```

## Model Downloads

Coming soon!

## Model Training

All configurations for the experiments are managed by [hydra](https://hydra.cc/), stored in `./config`.

Below you can find the list of experiments conducted in the paper:

- Small synthetic datasets: Planar and SBM introduced by [SPECTRE](https://arxiv.org/abs/2204.01613).
- Large graph datasets: Proteins and Point Clouds introduced by [GRAN](https://arxiv.org/abs/1910.00760).
- Molecular graph datasets: [QM9](https://arxiv.org/abs/1703.00564), [MOSES](https://github.com/molecularsets/moses), and [GuacaMol](https://github.com/BenevolentAI/guacamol).
- Our pre-training dataset (unattributed graphs): NetworkX, which is based on graph generators from [NetworkX](https://networkx.org/documentation/stable/reference/generators.html).

```bash
# You can replace planar with any of the above datasets
python train.py experiment=planar # can be sbm, protein, point_cloud, qm9, moses, guacamol, networkx
```


