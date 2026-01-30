# Single-cell RNA-seq + PyTorch baseline (PBMC3k)

This mini-project demonstrates an end-to-end single-cell RNA-seq analysis and deep learning workflow using Python and PyTorch. The project covers standard preprocessing and clustering of single-cell transcriptomics data with Scanpy, including normalization, highly-variable gene selection, PCA, Leiden clustering, and UMAP visualization. Building on these representations, multiple PyTorch-based classification baselines are implemented and compared, using both low-dimensional PCA embeddings and high-dimensional gene-expression features, with regularization and early stopping to assess generalization.

It emphasizes reproducible workflows, clear evaluation, and model checkpointing, reflecting common practices in applied machine learning research for biological data.

## What it does
- Loads PBMC3k single-cell RNA-seq data (Scanpy)
- Preprocesses: filtering, normalization, log1p, HVG selection
- Computes PCA + neighbors graph, performs Leiden clustering
- Visualizes clusters with UMAP (saved to `figures/`)
- Trains a PyTorch classifier on PCA embeddings to predict cluster IDs
- Evaluates with a confusion matrix (saved to `figures/`)
- Saves trained model weights to `models/`
- ## Baselines
    - PCA baseline: trains a PyTorch classifier on 50-dimensional PCA embeddings (test acc ~0.91)
    - HVG baseline: trains a PyTorch classifier directly on 2000 highly-variable gene features with dropout (test acc up to ~0.91; shows overfitting behavior on small data)
    - HVG + early stopping: trained a PyTorch classifier on highly-variable genes with dropout and early stopping (best test acc ~0.91)


## How to run
```bash
pip install numpy pandas matplotlib scikit-learn torch scanpy leidenalg
python pbmc_scanpy_pytorch_baseline.py
```

## Results

1. ### UMAP clustering
    ![UMAP clustering of PBMC cells](figures/umap_pbmc_clusters.png)
2. ### Confusion matrix — PCA baseline
    ![Confusion matrix PCA](figures/confusion_matrix.png)
3. models/pbmc_cluster_classifier.pt
4. ### Confusion matrix — HVG baseline
    ![Confusion matrix HVG](figures/confusion_matrix_hvg.png)
5. models/pbmc_hvg_cluster_classifier.pt
6. models/pbmc_hvg_classifier_earlystop_best.pt
7. ### Confusion matrix — HVG with early stopping
    ![Confusion matrix HVG early stopping](figures/confusion_matrix_hvg_earlystop.png)