from typing import Any, Dict, Tuple

import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.relax.frontend.nn.llm.position_embedding import switch_rope_freq_func

def _var_cpu(dtype):
    return T.alloc_buffer((1,), dtype)

def _rope(
    buffer: T.Buffer,
    offset: tir.Var,
    rotary_dim: int,
    theta: tir.Var,
    scale: tir.Var,
    indices: Tuple[tir.Var, ...],
    qkv_dtype: str,
    rope_scaling: Dict[str, Any],
):
    d = indices[-1]
    cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
        offset * scale, d, rotary_dim, theta, "float32"
    )
    cos = cos_freq * buffer[indices].astype("float32")
    sin = sin_freq * tir.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    ).astype("float32")
    expr = (cos + sin).astype(qkv_dtype)
    for var, value in var_map.items():
        expr = tir.Let(var, value, expr)
    return expr

def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    # pos if pos < sink_size else pos - sink_size + sliding_window_offset
    return T.if_then_else(
        pos < length_info[2, seq_id],
        pos,
        pos - length_info[2, seq_id] + length_info[1, seq_id],
    )

def _get_kv_chunk_len(num_pages, page_size, seq_id, length_info, sliding_window):
    if not sliding_window:
        return (num_pages - 1) * page_size + length_info[seq_id]
    # ((num_pages - 1) * page_size + last_page_len) - sliding_window_offset + sink_size
    return (
        (num_pages - 1) * page_size
        + length_info[0, seq_id]
        - length_info[1, seq_id]
        + length_info[2, seq_id]
    )

def _causal_mask(causal, row, col, kv_len, qo_len):
    return T.if_then_else(
        causal > 0,
        col < kv_len - qo_len + row + 1,
        col < kv_len,
    )

def _declare_length_info(var_length_info, batch_size, sliding_window, elem_offset):
    return (
        T.match_buffer(var_length_info, (3, batch_size), "int32", elem_offset=elem_offset)
        if sliding_window
        else T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=elem_offset)
    )

def _check_tree_order(tree_order_indptr, tree_order, batch, row, col, kv_len, qo_len):
    tree_order_len = tree_order_indptr[batch + 1] - tree_order_indptr[batch]

    tree_start = kv_len - tree_order_len
    child_idx_in_tree = row + tree_order_len - qo_len
    parent_idx_in_tree = col - tree_start
    return tir.all(
        col < kv_len,
        tir.any(
            col < tree_start,
            tir.all(
                tree_order[tree_order_indptr[batch] + child_idx_in_tree, 0]
                >= tree_order[tree_order_indptr[batch] + parent_idx_in_tree, 0],
                tree_order[tree_order_indptr[batch] + child_idx_in_tree, 0]
                < tree_order[tree_order_indptr[batch] + parent_idx_in_tree, 1],
            ),
        ),
    )