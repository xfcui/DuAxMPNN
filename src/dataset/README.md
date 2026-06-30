# Dataset and Dataloader for Training

This directory contains the dataset and dataloader implementation for PCQM4Mv2 graph training.

## Overview

The dataset is designed to efficiently load preprocessed molecular graphs from HDF5 files and batch them into a "collapsed" format suitable for JAX/Equinox training.

### Key Components

- `PCQMDataset`: Handles loading from HDF5, managing splits, and "collapsing" multiple graphs into a single batched graph block.
- `PCQMDataloader`: A simple iterator that yields batched graph blocks from a `PCQMDataset`.
- `batch_collapse`: The core logic for merging multiple molecular graphs into one large graph with a prepended "null" graph for padding.
- `dataprocess.py`, `features.py`, `graph.py`, `hdf5.py`: Preprocessing pipeline from raw SMILES/SDF to compact HDF5 tensors.

## Usage

### 1. Initialize the Dataset

By default, it looks for data in `Path.cwd() / "dataset" / "pcqm4m-v2"`. The dataset uses `h5py` to access data on disk, so it does not load the entire feature matrix into memory. Metadata like `node_ptr` and `labels` are loaded as NumPy arrays.

**Optimization**: The `PCQMDataset` class implements a shared cache. If you initialize multiple `PCQMDataset` instances pointing to the same data file, they will share the same metadata arrays and HDF5 file handle in memory.

```python
from src.dataset.dataset import PCQMDataset

# These two instances will share metadata in memory
train_dataset = PCQMDataset(split="train")
valid_dataset = PCQMDataset(split="valid")
```

### 2. Create a Dataloader

The `PCQMDataloader` provides a standard iteration interface.

```python
from src.dataset.dataset import PCQMDataset
from src.dataset.dataloader import PCQMDataloader

# Create dataset instances (metadata is shared)
train_dataset = PCQMDataset(split="train")
valid_dataset = PCQMDataset(split="valid")

# Create loaders
train_loader = PCQMDataloader(train_dataset, batch_size=256, shuffle=True)
valid_loader = PCQMDataloader(valid_dataset, batch_size=256)

for batch in train_loader:
    # batch is a dictionary of jnp.ndarray
    node_feat = batch["node_feat"]  # [total_nodes, feat_dim]
    edge_index = batch["edge_index"] # [2, total_edges]
    labels = batch["labels"]         # [batch_size]
    # ...
```

### 3. Collapsed Batch Format

To support JIT-compiled JAX functions with fixed shapes, we use a "collapsed" batching strategy:
- Multiple graphs are concatenated into a single large set of nodes and edges.
- A **null graph** is prepended to the batch to act as padding.
- The total number of nodes and edges is padded to a multiple of `PAD_TO_MULTIPLE` (default 1024).

#### Batch Dictionary Keys:
- `node_feat`: Node features with offsets applied.
- `node_embd`: Continuous node embeddings (if available).
- `node_ptr`: Pointers to the start of each graph in the node array.
- `node_batch`: Graph index for each node.
- `edge_index`: Edge indices (source, target).
- `edge_feat`: Edge features with offsets applied.
- `labels`: Training labels for the molecules in the batch.
- `molecule_ids`: Original dataset indices for the molecules.

Additional edge types (e.g., `edge_2hop_index`, `edge_2hop_feat`) are included if present in the processed data.

## Feature Offsets

Node and edge features are stored as raw category indices. `PCQMDataset` automatically applies offsets to these indices so they can be used directly with a single large embedding table (e.g., `node_feat_total_vocab`).
