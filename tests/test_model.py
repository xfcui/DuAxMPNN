import sys
from pathlib import Path

# Add project root and src/ to sys.path to ensure local imports resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import pytest
from src.model import get_model

def test_model_forward():
    # Set up model
    key = jax.random.PRNGKey(0)
    init_key, run_key = jax.random.split(key)
    
    # get_model defaults: depth=5, width=256, num_head=16, dim_head=16
    model = get_model(key=init_key)
    
    # Create a dummy collapsed batch of 2 graphs (excluding the null graph)
    # Total nodes: N = 6 (node 0 is null graph padding, nodes 1,2 for graph 1, nodes 3,4,5 for graph 2)
    N = 6
    B = 2
    
    node_feat = jnp.zeros((N, 10), dtype=jnp.int32)
    node_embd = jnp.zeros((N, 17), dtype=jnp.float32)
    node_batch = jnp.array([0, 1, 1, 2, 2, 2], dtype=jnp.int32)
    
    # 1-hop edges: let's connect node 1-2 (graph 1) and node 3-4, 4-5 (graph 2)
    # For null graph (index 0), no edges.
    edge_index = jnp.array([[1, 2, 3, 4, 4, 5], [2, 1, 4, 3, 5, 4]], dtype=jnp.int32)
    edge_feat = jnp.zeros((edge_index.shape[1], 6), dtype=jnp.int32)
    edge_batch = jnp.array([1, 1, 2, 2, 2, 2], dtype=jnp.int32)
    
    # Mock 2-hop, 3-hop, 4-hop edges
    edge_2hop_index = jnp.zeros((2, 0), dtype=jnp.int32)
    edge_2hop_feat = jnp.zeros((0, 2), dtype=jnp.int32)
    edge_2hop_batch = jnp.zeros((0,), dtype=jnp.int32)
    
    edge_3hop_index = jnp.zeros((2, 0), dtype=jnp.int32)
    edge_3hop_feat = jnp.zeros((0, 3), dtype=jnp.int32)
    edge_3hop_batch = jnp.zeros((0,), dtype=jnp.int32)
    
    edge_4hop_index = jnp.zeros((2, 0), dtype=jnp.int32)
    edge_4hop_feat = jnp.zeros((0, 4), dtype=jnp.int32)
    edge_4hop_batch = jnp.zeros((0,), dtype=jnp.int32)
    
    batch = {
        "node_feat": node_feat,
        "node_embd": node_embd,
        "node_batch": node_batch,
        "batch_n_graphs": B,
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "edge_batch": edge_batch,
        "edge_2hop_index": edge_2hop_index,
        "edge_2hop_feat": edge_2hop_feat,
        "edge_2hop_batch": edge_2hop_batch,
        "edge_3hop_index": edge_3hop_index,
        "edge_3hop_feat": edge_3hop_feat,
        "edge_3hop_batch": edge_3hop_batch,
        "edge_4hop_index": edge_4hop_index,
        "edge_4hop_feat": edge_4hop_feat,
        "edge_4hop_batch": edge_4hop_batch,
    }
    
    # Forward pass
    out = model(batch, key=run_key)
    
    # Output should have shape (B, 1)
    assert out.shape == (B, 1)
    assert not jnp.any(jnp.isnan(out))
