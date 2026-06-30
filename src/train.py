"""Training loop for PCQM4Mv2."""

import sys
from pathlib import Path

# Add project root and src directories to path so imports work both when
# running this file directly and via `python -m src.train`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings
# Suppress pkg_resources deprecation warning from outdated package (dependency of ogb)
warnings.filterwarnings('ignore', category=UserWarning, module='outdated')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from dataset import PCQMDataset, PCQMDataloader
from model import get_model

_H5_FOR_H_MODE = {
    "active": "data_processed.h5",
    "heavy": "data_processed_heavy.h5",
    "all": "data_processed_all.h5",
}

from optim import (
    _add_lora_product_decay,
    _add_scaled_decayed_weights,
    _make_lr_schedule,
    get_scheduled_hparams,
    lr_multiplier_for_param_path,
    make_optimizer,
    per_param_lr_multiplier_tree,
    per_param_wd_multiplier_tree,
    wd_multiplier_for_param_path,
)


def _resolve_dataset_root(hdf5_path: str | Path) -> Path:
    """Resolve dataset root from legacy `processed`-path-style input."""
    base = Path(hdf5_path)
    if base.name == "processed":
        return base.parent
    if base.name == "pcqm4m-v2":
        return base
    if (base / "processed").is_dir():
        return base
    return base.parent


def get_jax_dataloader(
    hdf5_path: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    seed: int | None = None,
    processed_h5: str = "data_processed.h5",
):
    """Instantiate a ``PCQMDataloader`` for the given split and batch size."""
    dataset_root = _resolve_dataset_root(hdf5_path)
    dataset = PCQMDataset(dataset_root=dataset_root, split=split, processed_h5=processed_h5)
    return PCQMDataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def to_jax_batch(batch):
    """Convert a dataloader batch to JAX arrays, keeping ``batch_n_graphs`` as a Python int."""
    converted = {}
    for k, v in batch.items():
        if k == "batch_n_graphs":
            converted[k] = int(v)
        else:
            converted[k] = jnp.asarray(v)
    return converted


def _check_nan_loss(x: jax.Array) -> None:
    """Host callback: raise if loss is non-finite."""
    if jnp.isnan(x):
        raise ValueError("Loss is NaN")


def loss_fn(model, batch, key, threshold=6e-2):
    """MAE loss between model predictions and labels; key=None runs deterministic inference."""
    preds = model(batch, training=(key is not None), key=key)
    preds = preds.squeeze(-1)  # (B, 1) -> (B,)
    loss = jnp.mean(jnp.abs(preds - batch["labels"]))
    jax.debug.callback(_check_nan_loss, loss)
    return loss


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    # scale_by_trust_ratio requires params to have the same tree structure as updates;
    # filter to arrays only, matching how opt_state was initialised.
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, batch):
    """Evaluation without gradients; key=None disables dropout for deterministic inference."""
    return loss_fn(model, batch, key=None)


def _train_one_epoch(
    model,
    optimizer,
    opt_state,
    train_loader,
    train_key: jax.Array,
    epoch: int,
    steps_per_epoch: int,
    scheduler_period: int | None,
    learning_rate: float,
    weight_decay: float,
):
    """Run one full training epoch and return updated model, opt state, key, and avg loss."""
    total_train_loss = 0.0
    num_train_batches = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(train_pbar):
        batch = to_jax_batch(batch)
        train_key, step_key = jax.random.split(train_key)
        model, opt_state, loss = train_step(model, opt_state, batch, optimizer, step_key)
        loss_val = loss.item()
        total_train_loss += loss_val
        num_train_batches += 1
        epoch_frac = (epoch * steps_per_epoch + batch_idx) / steps_per_epoch
        if scheduler_period is not None:
            cur_lr, cur_wd = get_scheduled_hparams(
                epoch_frac, scheduler_period, learning_rate, weight_decay
            )
        else:
            cur_lr, cur_wd = learning_rate, weight_decay
        train_pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{cur_lr:.2e}", wd=f"{cur_wd:.2e}")
    avg_train_loss = total_train_loss / max(num_train_batches, 1)
    return model, opt_state, train_key, avg_train_loss


def _validate_one_epoch(model, valid_loader, epoch: int) -> float:
    """Run one full validation epoch and return the average MAE loss."""
    total_valid_loss = 0.0
    num_valid_batches = 0
    valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]")
    for batch in valid_pbar:
        loss = eval_step(model, to_jax_batch(batch))
        loss_val = loss.item()
        total_valid_loss += loss_val
        num_valid_batches += 1
        valid_pbar.set_postfix(loss=f"{loss_val:.4f}")
    return total_valid_loss / max(num_valid_batches, 1)


def train(
    num_epochs=1,
    batch_size=32,
    learning_rate=1e-2,
    weight_decay=1e-2,
    model_save_path="results/best_model.eqx",
    scheduler_period=None,
    seed: int = 0,
    *,
    processed_h5: str = "data_processed.h5",
):
    """
    Train the GNN model on PCQM4Mv2 dataset.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (peak learning rate if using warmup schedule)
        weight_decay: L2 regularisation coefficient.
        model_save_path: Path to save the best model. Default: "results/best_model.eqx".
        scheduler_period: Period k for geometric LR scheduler. If None, use constant LR.
        seed: RNG seed for model init, training minibatch order (via dataloader), and dropout.
        processed_h5: Basename of the processed HDF5 under ``<root>/processed/``.

    Returns:
        Trained model
    """
    hdf5_path = "dataset/pcqm4m-v2/processed"
    train_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        processed_h5=processed_h5,
    )
    valid_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='valid',
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
        processed_h5=processed_h5,
    )

    if len(train_loader) == 0 or len(valid_loader) == 0:
        proc_path = _resolve_dataset_root(hdf5_path) / "processed" / Path(processed_h5).name
        raise RuntimeError(
            "Train or validation dataloader is empty (no graphs with label >= 0 in that split). "
            "This usually means the processed HDF5 has missing targets (e.g. all labels are -1). "
            f"Expected file: {proc_path}"
        )

    steps_per_epoch = len(train_loader)
    if scheduler_period is not None:
        lr_sched = _make_lr_schedule(steps_per_epoch, scheduler_period, learning_rate)
    else:
        lr_sched = learning_rate

    # Initialize model
    key = jax.random.PRNGKey(int(seed))
    model_key, train_key = jax.random.split(key)
    model = get_model(model_key)

    params = eqx.filter(model, eqx.is_array)
    lr_mult_tree = per_param_lr_multiplier_tree(params)
    wd_mult_tree = per_param_wd_multiplier_tree(params)

    print(f"Using Adan optimizer with lr={learning_rate} and wd={weight_decay}")
    optimizer = make_optimizer(
        learning_rate=lr_sched,
        weight_decay=weight_decay,
        wd_multiplier_tree=wd_mult_tree,
        lr_multiplier_tree=lr_mult_tree,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    print()

    # Track best validation loss
    best_valid_loss = float('inf')
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model, opt_state, train_key, avg_train_loss = _train_one_epoch(
            model, optimizer, opt_state, train_loader, train_key, epoch,
            steps_per_epoch, scheduler_period, learning_rate, weight_decay,
        )
        avg_valid_loss = _validate_one_epoch(model, valid_loader, epoch)

        msg = f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}"
        if avg_valid_loss < best_valid_loss:
            eqx.tree_serialise_leaves(model_save_path, model)
            best_valid_loss = avg_valid_loss
            msg += " *"
        print(f"\r{msg}")

    print(f"Training complete. Best validation loss: {best_valid_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--scheduler_period', type=int, default=8, help='Period for geometric LR scheduler')
    parser.add_argument('--model_save_path', type=str, default="results/best_model.eqx", help='Path to save the best model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for init, training shuffle, and dropout')
    parser.add_argument('--h-mode', type=str, default='active', choices=tuple(_H5_FOR_H_MODE.keys()))
    parser.add_argument('--processed-h5', type=str, default=None, help='Override processed HDF5 basename')
    args = parser.parse_args()

    processed_h5 = args.processed_h5 or _H5_FOR_H_MODE[args.h_mode]

    train(
        num_epochs=args.scheduler_period*8,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_save_path=args.model_save_path,
        scheduler_period=args.scheduler_period,
        seed=args.seed,
        processed_h5=processed_h5,
    )
