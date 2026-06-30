"""GNN model definitions for PCQM4Mv2: embeddings, conv/virt/depth/head kernels, and DuAxMPNN."""

import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax.ops import segment_sum

# dataset.py feature dimensions
# 10 atom features, 6/2/3/4 hop bond features; node_embd has 17 floats in HDF5, model uses 14 (RWPE12 + EN/GC)
from dataset import (
    NODE_FEAT_VOCAB_SIZES,
    NODE_FEAT_TOTAL_VOCAB,
    EDGE_FEAT_VOCAB_SIZES,
    EDGE_FEAT_TOTAL_VOCAB,
)

DROPOUT         = 0.1
EPSILON         = 1e-6
MINMAX_RATIO    = 20**.5
WIDTH_ACT_SCALE = 4

EMBED_POS  = 12    # RWPE12 only (ignore coord/en/geom auxiliaries)
EMBED_ELEC = 2
EDGE_SUFFIXES = list(EDGE_FEAT_VOCAB_SIZES.keys())
EDGE_DIMS_PER_HOP = [
    (EDGE_FEAT_TOTAL_VOCAB[suffix], len(EDGE_FEAT_VOCAB_SIZES[suffix]))
    for suffix in EDGE_SUFFIXES
]

# 1-hop edge storage: graph.py appends neighbor rank after bond_features (..., ring, rank) → rank at column 5.
_ONE_HOP_FEAT_SIZES = list(EDGE_FEAT_VOCAB_SIZES[""])
_ONE_HOP_OFFSETS = np.asarray(
    [1] + list(np.cumsum(_ONE_HOP_FEAT_SIZES[:-1], dtype=np.int32)),
    dtype=np.int32,
)
MAX_HOPS = 4
MOACT_BASES = 8
ELEC_CHANNELS = 4  # absolute mode: concat(src-dst, src+dst)


def _split_or_none(key, num):
    """Return a list of subkeys, or ``None`` if no key is provided."""
    if num == 0:
        return []
    if key is None:
        return [None] * num
    return list(jax.random.split(key, num))

def _inverse_softplus(y):
    """NumPy inverse softplus: ``x`` such that ``log(1 + exp(x)) == y`` (``y > 0``)."""
    return np.log(np.expm1(y))

def _count_params(model: eqx.Module) -> int:
    """Count the number of parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

def _clip_with_grad(x, min, max):
    """Clamp with gradients that flow through unclipped regions."""
    return jax.lax.stop_gradient(x.clip(min, max) - x) + x


class EmbedLayer(eqx.Module):
    """Memory-efficient multi-feature embedding unit."""
    embeddings: jnp.ndarray

    def __init__(self, total_vocab, num_features, width, key, *, init_std=1.0):
        total_dim = int(total_vocab)
        std = init_std / np.sqrt(num_features)
        self.embeddings = jax.random.normal(key, (total_dim, width)) * std

    def __call__(self, x):
        """Sum embedding lookups over the feature dimension."""
        return jnp.sum(self.embeddings[x], axis=-2)

# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(eqx.Module):
    """Learnable per-channel log-scale; forward multiplies by ``exp(scale)``."""
    scale: jnp.ndarray

    def __init__(self, width, scale_init=1.0):
        self.scale = jnp.full((width,), np.log(scale_init), dtype=jnp.float32)

    def __call__(self, x):
        s = jnp.exp(self.scale)
        s = _clip_with_grad(s, 1e-2, 1)
        return s * x

class LinearLayer(eqx.Module):
    """Bias-free linear projection with configurable init scale."""
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, key, *, init_std=1.0):
        std = init_std / np.sqrt(width_in)
        self.kernel = jax.random.normal(key, (width_in, width_out)) * std

    def __call__(self, x):
        return x @ self.kernel

class ActLayer(eqx.Module):
    """Softplus activation with learnable bias shift and optional inverted dropout."""
    bias: jnp.ndarray

    def __init__(self, width):
        self.bias = jnp.full((width,), _inverse_softplus(0.5), dtype=jnp.float32)

    def __call__(self, x, key=None):
        """Apply activation, clip output range, and drop units when ``key`` is given."""
        xx = jax.nn.softplus(x + self.bias)
        xx = _clip_with_grad(xx, 1/MINMAX_RATIO, MINMAX_RATIO)
        if key is not None and DROPOUT > 0.0:
            keep = 1.0 - DROPOUT
            mask = jax.random.bernoulli(key, p=keep, shape=xx.shape)
            xx = xx * mask.astype(xx.dtype) / keep
        return xx


class GroupLinearBlock(eqx.Module):
    """Groups-parallel linear: weight ``(num_head, d_in, d_out)``, no bias."""
    num_head: int
    dim_head: int
    linear: LinearLayer
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, num_head, dim_head, key):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.dim_head = dim_head
        self.linear = LinearLayer(width_in, width_norm, keys[0])
        assert width_out % num_head == 0
        self.kernel = jax.random.normal(keys[1], (num_head, dim_head, width_out // num_head)) / np.sqrt(dim_head)

    def __call__(self, x, norm_bias=None):
        """Project per-head with optional grouped normalization bias."""
        xx = self.linear(x)
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = xx / jnp.sqrt(jnp.mean(jnp.square(xx), axis=-1, keepdims=True) + EPSILON)
        if norm_bias is not None:
            xx = xx + norm_bias.reshape(xx.shape)
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        return xx

# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(eqx.Module):
    """Gated linear unit with group-linear and optional post-projection."""
    num_head: int
    dim_head: int
    lin_gat:  GroupLinearBlock
    lin_val:  GroupLinearBlock
    act_gat:  ActLayer
    kernel:   jnp.ndarray
    lin_out:  LinearLayer | None

    def __init__(self, width_in, width_out, num_head, dim_head, keep_groups=False, key=None, *, name=None):
        width_norm = num_head * dim_head
        width_act  = num_head * dim_head * WIDTH_ACT_SCALE
        keys = _split_or_none(key, 3 if keep_groups else 4)

        self.num_head = num_head
        self.dim_head = dim_head
        self.lin_gat  = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[0])
        self.lin_val = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[1])
        self.act_gat = ActLayer(width_act)
        if keep_groups:
            assert width_out % num_head == 0
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, width_out // num_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.lin_out = None
        else:
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, dim_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.lin_out = LinearLayer(width_norm, width_out, keys[3])

        if name is not None:
            print(f"##params[{name}]:", _count_params(self))

    def __call__(self, x, y=None, gate_bias=None, value_bias=None, key=None):
        """Run gate/value paths and combine them by elementwise product."""
        y = x if y is None else y

        gg = self.lin_gat(x, gate_bias)
        vv = self.lin_val(y, value_bias)
        xx = self.act_gat(gg, key) * vv
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        if self.lin_out is not None:
            xx = self.lin_out(xx)
        return xx


# Mixture of Activation — generalised from prelims/mix_tanh/README.md
_MOA_ACTS = {"tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid, "softplus": jax.nn.softplus}

class MoAct(eqx.Module):
    """Learnable embedding f(x)=Σ_i w_i·act(s_i·x) per channel, with user-chosen activation.

    Supported ``act``: ``"tanh"`` (bounded odd), ``"sigmoid"`` (bounded monotone),
    ``"softplus"`` (smooth relu-like, unbounded above).  Mixing weights are
    softplus-normalised and scales are softplus-positive.
    """

    num_channels: int = eqx.field(static=True)
    num_bases: int = eqx.field(static=True)
    act: str = eqx.field(static=True)
    weights: jnp.ndarray
    scales:  jnp.ndarray

    def __init__(self, num_channels: int, num_bases: int = 8, act: str = "tanh"):
        if act not in _MOA_ACTS:
            raise ValueError(f"act must be one of {list(_MOA_ACTS)}, got {act!r}")
        self.num_channels = num_channels
        self.num_bases = num_bases
        self.act = act

        lo = float(_inverse_softplus(0.1))
        hi = float(_inverse_softplus(10.0))
        self.weights = jnp.zeros((num_channels, num_bases))
        self.scales = jnp.tile(jnp.linspace(lo, hi, num_bases), (num_channels, 1))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        fn = _MOA_ACTS[self.act]
        w = jax.nn.softplus(self.weights)
        w = w / (w.sum(axis=-1, keepdims=True) + EPSILON)
        s = jax.nn.softplus(self.scales)
        return (fn(x[..., None] * s) * w).sum(-1)

class DiffEmbedLayer(eqx.Module):
    """Per-edge electronic embedding via MoAct → ``dim_head``."""

    num_channels: int = eqx.field(static=True)
    moa: MoAct
    linear: LinearLayer

    def __init__(self, num_channels, dim_head, key, *, init_std: float = 1.0):
        self.num_channels = int(num_channels)
        keys = _split_or_none(key, 1)
        self.moa = MoAct(num_channels, MOACT_BASES, act="tanh")
        self.linear = LinearLayer(num_channels, dim_head, keys[0], init_std=init_std)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(self.moa(x))


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvKernel(eqx.Module):
    """Bond-aware graph convolution with degree normalisation."""
    lora_down:  jnp.ndarray
    lora_up:    jnp.ndarray
    embed_edge: EmbedLayer
    embed_elec: DiffEmbedLayer
    embed_deg:  EmbedLayer
    lin_pre:    LinearLayer
    act_out:    GatedLinearBlock

    def __init__(self, num_head, dim_head, edge_total_vocab, edge_num_features, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 6)

        self.lora_down  = jax.random.normal(keys[4], (width_norm, dim_head)) / np.sqrt(width_norm)
        self.lora_up    = jnp.zeros((dim_head, width_norm), dtype=jnp.float32)
        self.embed_edge = EmbedLayer(edge_total_vocab, edge_num_features, dim_head, keys[0], init_std=0.01)
        self.embed_elec = DiffEmbedLayer(ELEC_CHANNELS, dim_head, keys[5], init_std=0.01)
        self.embed_deg  = EmbedLayer(6, 1, width_norm, keys[1], init_std=0.1)
        self.lin_pre    = LinearLayer(width_norm, width_norm, keys[2])
        self.act_out    = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[3])

        print(f"##params[conv]:", _count_params(self), edge_total_vocab, edge_num_features)

    def __call__(self, x, deg, edge_idx, edge_attr, node_elec, key=None):
        """Aggregate neighbor messages weighted by edge and degree embeddings."""
        src, dst = node_elec[edge_idx[0]], node_elec[edge_idx[1]]
        emb = jnp.concatenate([src - dst, src + dst], axis=-1)
        emb = self.embed_edge(edge_attr) + self.embed_elec(emb)
        msg = x[edge_idx[0]] + x[edge_idx[1]]
        msg = self.lin_pre(msg) + (msg @ self.lora_down * emb) @ self.lora_up
        msg = segment_sum(msg, edge_idx[1], len(x))
        msg = self.act_out(msg, gate_bias=self.embed_deg(deg[:, None]), key=key)
        return msg

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(eqx.Module):
    """Virtual node aggregation for global information exchange."""
    num_head: int
    act_out:  GatedLinearBlock

    def __init__(self, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.act_out  = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[1])

        print("##params[virt]:", _count_params(self))

    def __call__(self, x, batch, batch_size, key=None):
        """Pool node features to graph-level, transform, and broadcast update to each node."""
        msg = segment_sum(x, batch, batch_size)
        msg = self.act_out(msg, key=key)[batch]
        return msg


class SelfMixerKernel(eqx.Module):
    """Position-wise residual self-mixing block."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    act: GatedLinearBlock
    sca: ScaleLayer

    def __init__(self, width, num_head, dim_head, key=None):
        self.num_head = num_head
        self.dim_head = dim_head
        self.act = GatedLinearBlock(width, width, num_head, dim_head, key=key)
        self.sca = ScaleLayer(width, scale_init=1.0)

        print("##params[self_mixer]:", _count_params(self))

    def __call__(self, x, key=None):
        return self.sca(x) + self.act(x, key=key)

class LayerMixerKernel(eqx.Module):
    """Mixes k-hop convolutions and virtual node information."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    lin_pre:  GroupLinearBlock
    act_virt: VirtKernel
    lin_out:  LinearLayer
    sca_out:  ScaleLayer
    conv: tuple

    def __init__(self, width, num_head, dim_head, edge_dims_per_hop, key=None):
        num_hops = len(edge_dims_per_hop)
        width_norm = num_head * dim_head
        keys = _split_or_none(key, num_hops + 3)

        self.num_head = num_head
        self.dim_head = dim_head
        self.lin_pre  = GroupLinearBlock(width, width_norm, num_head, dim_head, keys[0])
        self.conv = tuple(
            ConvKernel(
                num_head,
                dim_head,
                edge_dims_per_hop[i][0],
                edge_dims_per_hop[i][1],
                key=keys[i + 3],
            )
            for i in range(num_hops)
        )
        self.act_virt = VirtKernel(num_head, dim_head, key=keys[1])
        self.lin_out = LinearLayer(width_norm, width, keys[2], init_std=1/(1 + (num_hops + 1) * .75**2)**.5)
        self.sca_out = ScaleLayer(width, scale_init=1.0)

        print("##params[layer_mixer]:", _count_params(self))

    def __call__(self, x, edges, batch, batch_size, node_elec, key=None):
        """Run multi-hop convolutions, mix with virtual-node broadcast, then residual-add."""
        keys = _split_or_none(key, len(self.conv) + 1)

        xx = self.lin_pre(x)
        for i, (conv, (idx, attr, deg)) in enumerate(zip(self.conv, edges)):
            xx = xx + conv(xx, deg, idx, attr, node_elec, key=keys[i])
        xx = xx + self.act_virt(xx, batch, batch_size, key=keys[-1])[batch]
        xx = self.sca_out(x) + self.lin_out(xx)
        return xx

class DepthMixerKernel(eqx.Module):
    """Cross-layer dense aggregation: projects all prior layer outputs and gates them into the current one."""
    num_head: int
    dim_head: int
    kernel:   jnp.ndarray | None
    act_out:  GatedLinearBlock

    def __init__(self, depth, width, num_head, dim_head, key=None):
        width_neck = dim_head * num_head // 4
        width_cat  = width + width_neck * depth
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.dim_head = dim_head
        if depth > 0:
            self.kernel = jax.random.normal(keys[0], (depth, width, width_neck)) / np.sqrt(width)
        else:
            self.kernel = None
        self.act_out = GatedLinearBlock(width_cat, width, num_head, dim_head, key=keys[1])

        print(f"##params[depth_mixer]:", _count_params(self), width_cat)

    def __call__(self, x, x_lst, key=None):
        """Gate current features against the sum of all previous layer outputs."""
        if self.kernel is None:
            xx = x
        else:
            xx = jnp.concatenate(x_lst, axis=-1)
            xx = xx.reshape(*x.shape[:-1], -1, x.shape[-1])
            xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
            xx = xx.reshape(*x.shape[:-1], -1)
            xx = jnp.concatenate([x, xx], axis=-1)
        xx = self.act_out(xx, key=key)
        return xx


class HeadKernel(eqx.Module):
    """Readout head: virtual + node pooling → scalar prediction."""
    num_head: int
    dim_head: int
    kernel:   jnp.ndarray
    act_out:  GatedLinearBlock
    readout_scale: jnp.ndarray
    readout_bias:  jnp.ndarray

    def __init__(self, depth, width, num_head, dim_head, key=None):
        width_neck = dim_head * num_head // 4
        width_cat  = width + width_neck * depth
        keys = _split_or_none(key, 3)

        self.num_head = num_head
        self.dim_head = dim_head
        if depth > 0:
            self.kernel = jax.random.normal(keys[0], (depth, width, width_neck)) / np.sqrt(width)
        else:
            self.kernel = jnp.zeros((0, width, width_neck), dtype=jnp.float32)
        self.act_out  = GatedLinearBlock(width_cat, 1, num_head*2, dim_head*2, key=keys[2])
        self.readout_scale = jnp.asarray(1.162127, dtype=jnp.float32)
        self.readout_bias  = jnp.asarray(5.689452, dtype=jnp.float32)

        print("##params[head]:", _count_params(self), width_cat)

    def __call__(self, x, x_lst, batch, batch_size, key=None):
        """Sum-pool nodes, fuse virtual node, project to scalar, and apply output affine."""
        if self.kernel.shape[0] == 0:
            xx = x
        else:
            xx = jnp.concatenate(x_lst, axis=-1)
            xx = xx.reshape(*x.shape[:-1], -1, x.shape[-1])
            xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
            xx = xx.reshape(*x.shape[:-1], -1)
            xx = jnp.concatenate([xx, x], axis=-1)
        yy = segment_sum(xx, batch, batch_size)
        yy = self.act_out(yy, key=key)
        yy = yy * self.readout_scale + self.readout_bias
        return yy


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
# DenseNet: https://arxiv.org/abs/1608.06993
# AttnRes: https://arxiv.org/abs/2603.15031
class DuAxMPNN(eqx.Module):
    """DuAxMPNN for PCQM4Mv2: 10 discrete atom features, multi-hop edges, RWPE12 + electronic node slice."""
    depth: int
    width: int
    num_head: int
    dim_head: int

    # Multi-feature atom encoder: one embedding table per atom feature dimension
    atom_embed: EmbedLayer
    atom_pos:   GatedLinearBlock
    layer_mix:  tuple
    depth_mix:  tuple
    head: HeadKernel

    def __init__(self, depth, width, num_head, dim_head, key):
        assert key is not None
        assert depth >= 1

        n_sub = depth * 2 + 3
        keys = _split_or_none(key, n_sub)

        self.depth = depth
        self.width = width
        self.num_head = num_head
        self.dim_head = dim_head
        edge_dims = EDGE_DIMS_PER_HOP[:MAX_HOPS]
        head_cross_depth = depth - 1

        print(
            f"#model={self.__class__.__name__}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_head={self.num_head}, dim_head={self.dim_head}, "
            f"max_hops={MAX_HOPS}, depth_mode=dense, cont_embed=moact, elec_mode=absolute"
        )

        curr = 0
        self.atom_embed = EmbedLayer(NODE_FEAT_TOTAL_VOCAB, len(NODE_FEAT_VOCAB_SIZES), width, keys[curr])
        curr += 1
        self.atom_pos = GatedLinearBlock(EMBED_POS, width, num_head, dim_head, 1, key=keys[curr])
        curr += 1
        layer_mix: list[LayerMixerKernel] = []
        depth_mix: list[DepthMixerKernel] = []
        for i in range(depth):
            layer_mix.append(
                LayerMixerKernel(width, num_head, dim_head, edge_dims, key=keys[curr])
            )
            curr += 1
            depth_mix.append(DepthMixerKernel(i, width, num_head, dim_head, key=keys[curr]))
            curr += 1
        self.layer_mix = tuple(layer_mix)
        self.depth_mix = tuple(depth_mix)
        self.head = HeadKernel(head_cross_depth, width, num_head, dim_head, key=keys[curr])

        print("#params:", _count_params(self))
        print()

    def __call__(self, batch, training=False, key=None):
        """Forward pass over a padded batch dict produced by the dataloader.

        Expected keys: ``node_feat`` (N_pad, 10), ``node_embd`` (N_pad, 17+),
        ``edgeX_{feat,index,batch}`` for X in ``{"", "_2hop", "_3hop", "_4hop"}``,
        ``node_batch`` (N_pad,), ``batch_n_graphs`` (scalar); graph index 0 is null.
        """
        node_feat = batch["node_feat"]
        node_embd = batch["node_embd"][..., :EMBED_POS]
        node_elec = batch["node_embd"][..., -EMBED_ELEC:]
        graph_id = batch["node_batch"]
        batch_size = batch["batch_n_graphs"] + 1
        edges = self._get_edge(batch)

        keys = _split_or_none(key if training else None, self.depth * 2 + 1)

        print(
            "#kernel: nodes={}, {}".format(
                node_feat.shape[0],
                ", ".join(
                    "{}_edges={}".format(
                        "1hop" if suffix == "" else suffix[1:],
                        batch[f"edge{suffix}_index"].shape[1],
                    )
                    for suffix in EDGE_SUFFIXES[:MAX_HOPS]
                ),
            )
        )

        x_depth = self.atom_embed(node_feat) + self.atom_pos(node_embd) / 10
        lst_layer: list[jax.Array] = []
        lst_depth: list[jax.Array] = []
        for i in range(self.depth):
            k_layer = keys[i * 2]
            k_depth = keys[i * 2 + 1]
            x_layer = self.layer_mix[i](x_depth, edges, graph_id, batch_size, node_elec, key=k_layer)
            lst_layer.append(x_layer)
            x_depth = self.depth_mix[i](x_layer, lst_layer[:-1], key=k_depth)
            lst_depth.append(x_depth)
        y = self.head(x_depth, lst_depth[:-1], graph_id, batch_size, key=keys[-1])[1:]
        return y

    def _get_edge(self, batch):
        """Build edge tuples ``(edge_index, edge_attr, degree)`` per hop."""
        num_nodes = batch["node_feat"].shape[0]
        edges = []
        for suffix in EDGE_SUFFIXES[:MAX_HOPS]:
            edge_index = batch[f"edge{suffix}_index"]
            edge_attr = batch[f"edge{suffix}_feat"]
            n_edges = batch[f"edge{suffix}_batch"].shape[0]
            deg = segment_sum(
                jnp.ones((n_edges, 1), dtype=edge_index.dtype),
                edge_index[1],
                num_nodes,
            ).squeeze(-1).clip(1, None)
            edges.append((edge_index, edge_attr, deg))
        return edges


def get_model(key):
    """Create DuAxMPNN (depth=5, width=256, heads=16)."""
    assert key is not None
    return DuAxMPNN(depth=5, width=256, num_head=16, dim_head=16, key=key)
