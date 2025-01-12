import enum
import math
import tvm
from typing import Any, Dict, Tuple

from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.target import Target

from tvm.relax.frontend.nn.llm.position_embedding import llama_rope_with_position_map, switch_rope_freq_func


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


def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    # pos if pos < sink_size else pos - sink_size + sliding_window_offset
    return T.if_then_else(
        pos < length_info[2, seq_id],
        pos,
        pos - length_info[2, seq_id] + length_info[1, seq_id],
    )

def _declare_length_info(var_length_info, batch_size, sliding_window, elem_offset):
    return (
        T.match_buffer(var_length_info, (3, batch_size), "int32", elem_offset=elem_offset)
        if sliding_window
        else T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=elem_offset)
    )

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

def _attention_decode_cpu(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    sliding_window: bool,
    rope_scaling: Dict[str, Any],
):
    log2e = math.log2(math.exp(1))
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim
    group_size = num_qo_heads //num_kv_heads

    global_symbol = "batch_decode_paged_kv"
    if sliding_window:
        global_symbol += "_sliding_window"

    @T.prim_func(check_well_formed=False)
    def batch_decode_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset_handle: T.handle,
        q_rope_position_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1, "global_symbol": global_symbol})
        B = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype) #query 值
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, 16, D), qkv_dtype
        ) # page_kv_cache 值
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32", elem_offset=page_indptr_elem_offset) # 紀錄每個batch的page起始 前減後 知道這個batch用了幾頁
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)# 哪些 page實際有存值
        k_rope_pos_offset = T.match_buffer(k_rope_pos_offset_handle, (B,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(q_rope_position_handle, (B,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(output_handle, (B, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (B, H_qo), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, B, sliding_window, length_info_elem_offset)

        sm_scale = 1.0 / math.sqrt(float(D)) * log2e

        for b in T.serial(B):
            with T.block("attn"):
                
                O_local = T.alloc_buffer((D, ), "float32")
                Q_local = T.alloc_buffer((D, ), "float32")
                K_local = T.alloc_buffer((D, ), "float32")
                V_local = T.alloc_buffer((D, ), "float32")

                #cur_page_indptr_begin = T.alloc_buffer([1, ], "int32")
                #cur_page_indptr_end = T.alloc_buffer((1, ), "int32")
                num_pages_for_b = T.alloc_buffer((1, ), "int32")
                kv_chunk_len = T.alloc_buffer((1, ), "int32")

                m_val = T.alloc_buffer((1, ), "float32")
                new_m = T.alloc_buffer((1, ), "float32")
                d_val = T.alloc_buffer((1, ), "float32")
                S_val = T.alloc_buffer((1, ), "float32")
                scale_O = T.alloc_buffer((1, ), "float32")
                factor = T.alloc_buffer((1, ), "float32")

                cur_page_indptr_begin: T.int32 = page_table_indptr[b]
                cur_page_indptr_end: T.int32 = page_table_indptr[b + 1] 

                #num_pages_for_b[0] = page_table_indptr[b + 1] - page_table_indptr[b]
                kv_chunk_len[0] = T.if_then_else(
                    cur_page_indptr_begin != cur_page_indptr_end,
                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, b, length_info, sliding_window),
                    0
                )


                for h_qo in T.serial(H_qo):
                    
                    m_val[0] = -5e4
                    d_val[0] = 1.0

                    for d in T.serial(D):
                        O_local[d] = 0.0

                    for d in T.serial(D):
                        Q_local[d] = T.if_then_else(
                            rotary_mode == 1,
                            _rope(Q, q_rope_position[b], head_dim, rope_theta, rope_scale, (b, h_qo, d), qkv_dtype, rope_scaling),
                            Q[b, h_qo, d]
                        )

                    for row_idx in T.serial(kv_chunk_len[0]):
                        seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b, length_info, sliding_window)
                        page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + (seq_offset // 16)]
                        page_offset: T.int32(is_size_var=True) = seq_offset % 16

                        for d in T.serial(D):
                            K_local[d] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(pages, k_rope_pos_offset[b] + row_idx, head_dim, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d), qkv_dtype, rope_scaling),
                                pages[page_no, 0, h_qo // group_size, page_offset, d]
                            )
                        S_val[0] = 0.0
                        for d in T.serial(D):
                            S_val[0] += Q_local[d] * K_local[d]
                        
                        S_val[0] *= attn_score_scaling_factor * sm_scale

                        new_m[0] = T.max(m_val[0], S_val[0])
                        d_val[0] = (d_val[0] * T.exp2(m_val[0] - new_m[0])) + T.exp2(S_val[0] - new_m[0])

                        scale_O[0] = T.exp2(m_val[0] - new_m[0])

                        for d in T.serial(D):
                            O_local[d] = O_local[d] * scale_O[0]

                        m_val[0] = new_m[0]
                        for d in T.serial(D):
                            V_local[d] = pages[page_no, 1, h_qo // group_size, page_offset, d]

                        factor[0] = T.exp2(S_val[0] - m_val[0])
                        for d in T.serial(D):
                            O_local[d] = O_local[d] + V_local[d] * factor[0]
                    
                    for d in T.serial(D):
                        O_local[d] = O_local[d] / d_val[0]
                        output[b, h_qo, d] = O_local[d]
                    
                    lse[b, h_qo] = m_val[0] + T.log2(d_val[0])

    return batch_decode_paged_kv

