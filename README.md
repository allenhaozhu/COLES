##  Contrastive Laplacian Eigenmaps (COLES)

### Overview
This repo contains an example implementation of the Neurips 2021 submission: Contrastive Laplacian Eigenmaps.
This code is based on SSGC.
In this code, we provide codes for Table 2 and node clustering experiments. To prevent unnecessary trouble, we submit the data along with the code.
For reddits and ogb-arxiv, we do not provide the code because the corresponding is too big.
We also provide the log file for checking to avoid if the code cannot run correctly in some unknown situations.

This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed, Cora Full).


### Dependencies
Our implementation works with PyTorch>=1.0.0 

### Data
We provide the citation network datasets under `data/`.

### Usage

```
$ python train_ssgc_(dataset_name).py
$ python train_ssgc_(dataset_name)_clustering.py
```

# COLES
