import enum
import math
from typing import Any, Dict, Tuple

import tvm
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.script import ir as I
from tvm.target import Target

from tvm.relax.frontend.nn.llm.position_embedding import llama_rope_with_position_map, switch_rope_freq_func

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


def tree_attn_cpu(
    h_kv, h_q, d, dtype, rope_scaling: Dict[str, Any]
):
    
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    @T.prim_func
    def batch_tree_attn(  # pylint: disable=too-many-branches
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_k: T.handle, # [total_len, h_kv, d]
        var_v: T.handle, # [total_len, h_kv, d]
        var_kv_indptr: T.handle, # [batch_size + 1], kv_indptr should be the same as q_indptr in this case
        var_q_rope_position: T.handle, # [total_q_len]
        var_mn_indptr: T.handle, # [batch_size + 1]
        var_mask: T.handle, # [mn_indptr[batch_size]]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32,
        batch_size: T.int32,
    ):
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        mn_indptr_elem_offset = T.int32(is_size_var=True)
        mask_elem_offset = T.int32(is_size_var=True)
        tree_size = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        mn_indptr = T.match_buffer(var_mn_indptr, (batch_size + 1,), "int32", elem_offset=mn_indptr_elem_offset)
        mask = T.match_buffer(var_mask, (tree_size, 2), "int32", elem_offset=mask_elem_offset)
        output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        for b in T.serial(batch_size):
                with T.block("attn"):
                    
                    # q_token_start = T.alloc_buffer([1,], "uint32")
                    # q_num = T.alloc_buffer([1,], "uint32")
                    # k_token_start = T.alloc_buffer([1,], "int32")
                    # k_num = T.alloc_buffer([1,], "int32")

                    softmax_sum = T.alloc_buffer([h_q], "float32")
                    m_prev = T.alloc_buffer([h_q], "float32")
                    m_new = T.alloc_buffer([h_q], "float32")
                    d_prev = T.alloc_buffer([h_q], "float32")
                    d_new = T.alloc_buffer([h_q], "float32")
                    sum = T.alloc_buffer([d], "float32")

                    

                    max_score = T.alloc_buffer([h_q], "float32")
                    attention_scores = T.alloc_buffer([kv_len, h_q], "float32")
                    exp_scores = T.alloc_buffer([kv_len, h_q], "float32")
                    attention_score = T.alloc_buffer([1,], "float32")
                    query_val = T.alloc_buffer([1,], "float32")
                    key_val = T.alloc_buffer([1,], "float32")
                    result = T.alloc_buffer([1,], "float32")

                    for q_idx in T.serial(q_indptr[b + 1] - q_indptr[b]):
                    
                        for i in T.serial(h_q):
                            max_score[i] = -5e4
                            m_prev[i] = -5e4
                            d_prev[i] = 1.0

                        for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            for h in T.serial(h_q):
                                h_kv_idx = h // group_size

                                if _check_tree_order(
                                                row=q_idx,
                                                col=k_idx,
                                                batch=b,
                                                tree_order=mask,
                                                tree_order_indptr=mn_indptr,
                                                kv_len=kv_indptr[b + 1] - kv_indptr[b],
                                                qo_len=q_indptr[b + 1] - q_indptr[b]):
                                    result[0] = 0.0
                                    for d_idx in T.serial(d):
                                        query_val[0] = T.if_then_else(
                                            rotary_mode == 1,
                                            _rope(q, q_rope_position[q_indptr[b] + q_idx], d, rope_theta, rope_scale, (q_indptr[b] + q_idx, h, d_idx), dtype, rope_scaling),
                                            q[q_indptr[b] + q_idx, h, d_idx]
                                        )

                                        key_val[0] = T.if_then_else(
                                            rotary_mode == 1,
                                            _rope(k, q_rope_position[kv_indptr[b] + k_idx], d, rope_theta, rope_scale, (kv_indptr[b] + k_idx, h_kv_idx, d_idx), dtype, rope_scaling),
                                            k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]
                                        )

                                        result[0] += query_val[0] * key_val[0]
                                        #result[0] += q[q_indptr[b] + q_idx, h, d_idx] * k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]
                                    attention_score[0] = result[0] * sm_scale * attn_score_scaling_factor
                                            
                                else:
                                    attention_score[0] = -5e4 * sm_scale * attn_score_scaling_factor
                                attention_scores[k_idx, h] = attention_score[0]
                                max_score[h] = T.max(max_score[h], attention_score[0])
                                m_new[h] = T.max(m_prev[h], max_score[h])
                                

                        for h in T.serial(h_q):
                            d_new[h] = d_prev[h] * T.exp2(m_prev[h] - m_new[h])                

                        for h in T.serial(h_q):
                            softmax_sum[h] = 0.0
                            for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                                exp_scores[k_idx, h] = T.exp2(attention_scores[k_idx, h] - m_new[h])
                                softmax_sum[h] += exp_scores[k_idx, h]
                            d_new[h]+=softmax_sum[h]
                                
                        d_prev = d_new
                        m_prev = m_new

                        for h in T.serial(h_q):
                            h_kv_idx = h // group_size
                            
                            for i in T.serial(d):
                                sum[i] = 0.0
                            for v_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                                weight = exp_scores[v_idx, h] / d_new[h]
                                for i in T.serial(d):
                                    sum[i] += v[kv_indptr[b] + v_idx, h_kv_idx, i] * weight
                            for i in T.serial(d):
                                output[q_indptr[b] + q_idx, h, i] = sum[i]
                            
                            lse[q_indptr[b] + q_idx, h] = m_prev[h] + T.log2(d_prev[h])
    return batch_tree_attn

