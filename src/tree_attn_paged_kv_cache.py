import math
from typing import Any, Dict
from tvm.script import tir as T
from tvm.target import Target

from .utils import _rope, _get_seq_offset, _get_kv_chunk_len, _causal_mask, _declare_length_info, _check_tree_order


def _tree_attn_paged_kv_cpu(
    h_kv, h_q, d, dtype, rope_scaling: Dict[str, Any], target: Target
):
    global_symbol = "tree_attn_paged_kv_cpu"
    sliding_window = False
    NUM_BLKS = 16
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))
    # pylint: disable=line-too-long,too-many-branches
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def tree_attn_paged_kv_cpu(
        _0: T.int32,  # pylint: disable=unused-argument
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, 2, h_kv, page_size, d]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        var_k_rope_pos_offset: T.handle, # [b]
        var_q_rope_position: T.handle, # [total_len]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32,
        tree_order_indptr_handle: T.handle,  # [batch_size + 1]
        tree_order_handle: T.handle,  # [total_len, 2]
    ):
        T.func_attr({"global_symbol": global_symbol}) 
        batch_size = T.int32(is_size_var=True) 
        total_len = T.int32(is_size_var=True) 
        nnz_pages = T.int32(is_size_var=True) 
        max_num_pages = T.int32(is_size_var=True) 
        q_indptr_elem_offset = T.int32(is_size_var=True) 
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)
        tree_order_elem_offset = T.int32(is_size_var=True)
        tree_order_indptr_elem_offset = T.int32(is_size_var=True)

        
        q = T.match_buffer(var_q, (total_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, 16, d), dtype)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable
        tree_order_indptr = T.match_buffer(
            tree_order_indptr_handle,
            (batch_size + 1,),
            "int32",
            elem_offset=tree_order_indptr_elem_offset,
        )
        total_tree_order_len = T.int32(is_size_var=True)
        tree_order = T.match_buffer(
            tree_order_handle,
            (total_tree_order_len, 2),
            "int32",
            elem_offset=tree_order_elem_offset,
        )
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, batch_size, sliding_window, length_info_elem_offset)


        T.Assert(
            rotary_mode == T.int32(0), "Inline rotary mode is not supported in tree attention."
        )

        for h_qo in T.serial(h_q):
            for b_idx in T.serial(batch_size):
                with T.block("attn"):
                    O_local = T.alloc_buffer((d, ), "float32")
                    Q_local = T.alloc_buffer((d, ), "float32")
                    K_local = T.alloc_buffer((d, ), "float32")
                    V_local = T.alloc_buffer((d, ), "float32")

                    kv_chunk_len = T.alloc_buffer((1, ), "int32")

                    m_val = T.alloc_buffer((1, ), "float32")
                    new_m = T.alloc_buffer((1, ), "float32")
                    d_val = T.alloc_buffer((1, ), "float32")
                    S_val = T.alloc_buffer((1, ), "float32")
                    scale_O = T.alloc_buffer((1, ), "float32")
                    factor = T.alloc_buffer((1, ), "float32")
                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1] 
                    #max_kv_len: T.int32 = max_num_pages * 16
                    kv_chunk_len[0] = T.if_then_else(
                        cur_page_indptr_begin != cur_page_indptr_end,
                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, b_idx, length_info, sliding_window),
                        0
                    )


                    for q_idx in T.serial(q_indptr[b_idx + 1] - q_indptr[b_idx]):
                        
                        #init m, d, O
                        m_val[0] = -5e4
                        d_val[0] = 1.0
                        for d_idx in T.serial(d):
                            O_local[d_idx] = 0.0
                        curl_q: T.int32 = q_indptr[b_idx] + q_idx

                        for d_idx in T.serial(d):

                            Q_local[d_idx] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(q, q_rope_position[curl_q], d, rope_theta, rope_scale, (curl_q, h_qo, d_idx), dtype, rope_scaling),
                                q[curl_q, h_qo, d_idx]
                            )
                        for row_idx in T.serial(max_num_pages * 16):
                            if row_idx < kv_chunk_len[0]:
                                # seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b_idx, length_info, sliding_window)
                                #seq_offset: T.int32(is_size_var=True) = row_idx
                                page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + (_get_seq_offset(row_idx, b_idx, length_info, sliding_window) // 16)]
                                page_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b_idx, length_info, sliding_window) % 16

                                # Load KV
                                for d_idx in T.serial(d):
                                    K_local[d_idx] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(pages, k_rope_pos_offset[b_idx] + row_idx, d, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d_idx), dtype, rope_scaling),
                                        pages[page_no, 0, h_qo // group_size, page_offset, d_idx]
                                    )
                                    V_local[d_idx] = pages[page_no, 1, h_qo // group_size, page_offset, d_idx]

                                # Compute S
                                # Q[i] * K[i]   * attn_score * sm_scale
                                S_val[0] = 0.0
                                for d_idx in T.serial(d):
                                    S_val[0] += Q_local[d_idx] * K_local[d_idx]
                                S_val[0] *= attn_score_scaling_factor * sm_scale

                                # update m_val, d_val , O_local
                                if _check_tree_order(
                                    tree_order_indptr=tree_order_indptr,
                                    tree_order=tree_order,
                                    batch=b_idx,
                                    row=q_idx,
                                    col=row_idx,
                                    kv_len=kv_chunk_len[0],
                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                ):
                                    new_m[0] = T.max(m_val[0], S_val[0])
                                else:
                                    S_val[0] = -5e4
                                # update d_val
                                d_val[0] *= T.exp2(m_val[0] - new_m[0])  
                                d_val[0] += T.exp2(S_val[0] - new_m[0])

                                # restore O_local then update O_local
                                scale_O[0] = T.exp2(m_val[0] - new_m[0])
                                m_val[0] = new_m[0]
                                factor[0] = T.exp2(S_val[0] - m_val[0])
                                for d_idx in T.serial(d):
                                    O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]


                                for d_idx in T.serial(d):
                                    O_local[d_idx] += V_local[d_idx] * factor[0]
                               
                        # Store Output 
                        for d_idx in T.serial(d):
                            O_local[d_idx] = O_local[d_idx] /d_val[0]
                            output[curl_q, h_qo, d_idx] = O_local[d_idx]
                        
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    return tree_attn_paged_kv_cpu