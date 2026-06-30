"""Compatibility module exporting the canonical dataset and dataloader APIs."""

from __future__ import annotations

from pathlib import Path
import importlib.util


def _load_submodule(module_name: str, relative_path: str):
    """Load a Python file as a named module, resolving its path relative to this file."""
    module_path = Path(__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATASET_MODULE = _load_submodule("_pcqm_dataset_core", "dataset/dataset.py")
_DATALOADER_MODULE = _load_submodule("_pcqm_dataset_dataloader", "dataset/dataloader.py")


PCQMDataset = _DATASET_MODULE.PCQMDataset
PCQMDataloader = _DATALOADER_MODULE.PCQMDataloader
batch_collapse = _DATASET_MODULE.batch_collapse

PAD_TO_MULTIPLE = _DATASET_MODULE.PAD_TO_MULTIPLE
NODE_FEAT_VOCAB_SIZES = _DATASET_MODULE.NODE_FEAT_VOCAB_SIZES
NODE_FEAT_TOTAL_VOCAB = _DATASET_MODULE.NODE_FEAT_TOTAL_VOCAB
EDGE_FEAT_VOCAB_SIZES = _DATASET_MODULE.EDGE_FEAT_VOCAB_SIZES
EDGE_FEAT_TOTAL_VOCAB = _DATASET_MODULE.EDGE_FEAT_TOTAL_VOCAB


__all__ = [
    "batch_collapse",
    "PCQMDataset",
    "PCQMDataloader",
    "PAD_TO_MULTIPLE",
    "NODE_FEAT_VOCAB_SIZES",
    "NODE_FEAT_TOTAL_VOCAB",
    "EDGE_FEAT_VOCAB_SIZES",
    "EDGE_FEAT_TOTAL_VOCAB",
]
