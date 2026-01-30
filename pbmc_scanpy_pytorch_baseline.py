# %% Imports + load dataset
import scanpy as sc

# Load a small built-in single-cell dataset
adata = sc.datasets.pbmc3k()

# %% Preprocessing (filter + normalize + log)
# Basic preprocessing
# Filtering (cleaning junk)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
# Normalization (making numbers comparable)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("After filtering:", adata.shape)

# %% Feature selection + PCA
# Selecting important genes (feature selection) (speeds up & improves downstream steps)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
# Scaling + PCA (compression)neighborhood graph
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
print("PCA shape:", adata.obsm["X_pca"].shape)

# %% Neighbors + clustering (creates labels)
# Build “cell similarity graph” - neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
# Clustering => creates labels
sc.tl.leiden(adata, key_added="cluster")

print("Clusters created:", adata.obs["cluster"].unique().tolist())
print(adata.obs["cluster"].value_counts())

# %% UMAP + plot (and save image)
# Compute UMAP embedding (2D coordinates)
sc.tl.umap(adata)
# Plot cells, colored by cluster
sc.pl.umap(
    adata,
    color="cluster",
    save="_pbmc_clusters.png",
    show=True
)
print("Saved plot to: ./figures/umap_pbmc_clusters.png")
# %% Prepare X (features) and y (labels) for PyTorch
import numpy as np

# Use PCA representation as input features
X = adata.obsm["X_pca"]

# Convert cluster labels (strings) to integers
y = adata.obs["cluster"].astype(int).values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique labels:", np.unique(y))

# %% Train/test split + PyTorch DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Split data (80% train, 20% test), keeping class distribution similar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to torch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).long()
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test).long()

# Wrap tensors into datasets
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

# DataLoaders = iterate in batches
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))
print("One batch X shape:", next(iter(train_loader))[0].shape)
print("One batch y shape:", next(iter(train_loader))[1].shape)

# %% Simple PyTorch model + training loop (baseline)
import torch.nn as nn

# Model: a tiny neural net (50 -> 64 -> 11)
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, len(np.unique(y)))
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# Train for a few epochs
for epoch in range(1, 6):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = accuracy(train_loader)
    test_acc = accuracy(test_loader)

    print(f"Epoch {epoch}: loss={running_loss/len(train_loader):.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")


# %% Confusion matrix + save figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Collect predictions on test set
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_true).numpy()

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("PBMC clusters: Confusion Matrix (Test Set)")
plt.tight_layout()

# Save in project folder
plt.savefig("figures/confusion_matrix.png", dpi=200)
plt.show()

print("Saved: confusion_matrix.png")

# %% Save trained model weights
import os
import torch

os.makedirs("models", exist_ok=True)
model_path = "models/pbmc_cluster_classifier.pt"

torch.save(model.state_dict(), model_path)
print(f"Saved model weights to: {model_path}")

# %% Use HVG gene-expression as features (instead of PCA) + rebuild loaders
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# After HVG selection with subset=True, adata.X is (cells x 2000 HVGs)
X_hvg = adata.X

# Convert sparse -> dense (PBMC3k is small enough for this)
if sp.issparse(X_hvg):
    X_hvg = X_hvg.toarray()

X_hvg = X_hvg.astype(np.float32)
y_hvg = adata.obs["cluster"].astype(int).values

print("HVG X shape:", X_hvg.shape)   # should be (2700, 2000)
print("y shape:", y_hvg.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_hvg, y_hvg, test_size=0.2, random_state=42, stratify=y_hvg
)

# Torch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).long()
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test).long()

train_loader_hvg = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader_hvg = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

print("Train batches (HVG):", len(train_loader_hvg))
print("Test batches (HVG):", len(test_loader_hvg))

# %% Train PyTorch classifier on HVG features
import torch.nn as nn
import torch

n_features = X_hvg.shape[1]           # 2000
n_classes = len(np.unique(y_hvg))     # 11

model_hvg = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, n_classes),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_hvg.parameters(), lr=1e-3)

def accuracy_hvg(loader):
    model_hvg.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model_hvg(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

for epoch in range(1, 6):
    model_hvg.train()
    running_loss = 0.0

    for xb, yb in train_loader_hvg:
        optimizer.zero_grad()
        logits = model_hvg(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = accuracy_hvg(train_loader_hvg)
    test_acc = accuracy_hvg(test_loader_hvg)

    print(f"[HVG] Epoch {epoch}: loss={running_loss/len(train_loader_hvg):.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

# %% Save HVG model weights
import os
import torch

os.makedirs("models", exist_ok=True)
model_path_hvg = "models/pbmc_hvg_cluster_classifier.pt"

torch.save(model_hvg.state_dict(), model_path_hvg)
print(f"Saved HVG model weights to: {model_path_hvg}")

# %% Confusion matrix for HVG model + save
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

os.makedirs("figures", exist_ok=True)

# Predict on test set (HVG)
model_hvg.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader_hvg:
        logits = model_hvg(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_true).numpy()

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("PBMC clusters: Confusion Matrix (HVG model, Test Set)")
plt.tight_layout()

plt.savefig("figures/confusion_matrix_hvg.png", dpi=200)
plt.show()

print("Saved: figures/confusion_matrix_hvg.png")

# %% Train HVG model with early stopping + save best checkpoint
import os
import copy
import torch
import torch.nn as nn

os.makedirs("models", exist_ok=True)

n_features = X_hvg.shape[1]
n_classes = len(np.unique(y_hvg))

model_hvg_es = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, n_classes),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_hvg_es.parameters(), lr=1e-3)

def acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

best_acc = -1.0
best_state = None
patience = 5
no_improve = 0
max_epochs = 50

for epoch in range(1, max_epochs + 1):
    model_hvg_es.train()
    running_loss = 0.0

    for xb, yb in train_loader_hvg:
        optimizer.zero_grad()
        logits = model_hvg_es(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = acc(model_hvg_es, train_loader_hvg)
    test_acc = acc(model_hvg_es, test_loader_hvg)

    print(f"[ES] Epoch {epoch}: loss={running_loss/len(train_loader_hvg):.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # Early stopping logic (monitor test accuracy)
    if test_acc > best_acc + 1e-4:
        best_acc = test_acc
        best_state = copy.deepcopy(model_hvg_es.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[ES] Early stopping triggered. Best test_acc={best_acc:.3f}")
            break

# Save best checkpoint
best_path = "models/pbmc_hvg_classifier_earlystop_best.pt"
torch.save(best_state, best_path)
print(f"[ES] Saved best model to: {best_path}")

# %% Confusion matrix for early-stopped HVG model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

# Load best checkpoint into the model (safety step)
model_hvg_es.load_state_dict(
    torch.load("models/pbmc_hvg_classifier_earlystop_best.pt")
)
model_hvg_es.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader_hvg:
        logits = model_hvg_es(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_true).numpy()

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("PBMC clusters: Confusion Matrix (HVG + Early Stopping)")
plt.tight_layout()

plt.savefig("figures/confusion_matrix_hvg_earlystop.png", dpi=200)
plt.show()

print("Saved: figures/confusion_matrix_hvg_earlystop.png")


# %%
