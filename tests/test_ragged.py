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


def _var(dtype):
    return T.alloc_buffer((1,), dtype, scope="local")

def _causal_mask(causal, row, col, kv_len, qo_len):
    return T.if_then_else(
        causal > 0,
        col < kv_len - qo_len + row + 1,
        col < kv_len,
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

def _attention_prefill_ragged_gpu(h_kv, h_q, d, dtype, rope_scaling: Dict[str, Any]):
    # pylint: disable=line-too-long
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16

    

    # fmt: off


    @I.ir_module(check_well_formed=False)
    class gpu_module:

        @T.prim_func
        def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
            var_q: T.handle, # [total_len, h_q, d]
            var_q_indptr: T.handle, # [batch_size + 1]
            var_k: T.handle, # [total_len, h_kv, d]
            var_v: T.handle, # [total_len, h_kv, d]
            var_kv_indptr: T.handle, # [batch_size + 1]
            var_q_rope_position: T.handle, # [total_q_len]
            var_k_rope_pos_offset: T.handle, # [b]
            var_output: T.handle, # [total_len, h_q, d]
            var_lse: T.handle, # [total_len, h_q]
            causal: T.int32,
            rotary_mode: T.int32,
            rope_scale: T.float32,
            rope_theta: T.float32,
            attn_score_scaling_factor: T.float32
        ):
            batch_size = T.int32(is_size_var=True)
            qo_len = T.int32(is_size_var=True)
            kv_len = T.int32(is_size_var=True)
            q_indptr_elem_offset = T.int32(is_size_var=True)
            kv_indptr_elem_offset = T.int32(is_size_var=True)
            q_rope_position_elem_offset = T.int32(is_size_var=True)
            k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

            q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
            q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
            k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
            v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
            kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
            q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
            k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
            output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
            lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable


         # # kernel code
            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                    for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                        for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                            with T.block("attn"):
                                bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                                T.reads()
                                T.writes()
                                tile_id = _var("int32")
                                batch_idx = _var("int32")
                                batch_tiles = _var("int32")
                                batch_rows = _var("int32")
                                iterator = _var("int32")
                                kv_chunk_len = _var("int32")

                                Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                                K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                                V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                                S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                                S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                                O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                                m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                                m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                                d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                                m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                                m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                                d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                                tile_id[0] = bx
                                batch_idx[0] = 0
                                batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                                batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                                while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                    while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                        tile_id[0] -= batch_tiles[0]
                                        batch_idx[0] += 1
                                        if batch_idx[0] < batch_size:
                                            b_idx: T.int32 = batch_idx[0]
                                            batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                    if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                        b_idx: T.int32 = batch_idx[0]
                                        q_indptr_val: T.int32 = q_indptr[b_idx]
                                        LH_start: T.int32 = tile_id[0] * tile_x

                                        kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                        T.tvm_storage_sync("shared")

                                    # init states
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                m_smem[row] = -5e4
                                                d_smem[row] = 1.0

                                        for li, lj in T.grid(tile_x, tile_y):
                                            with T.block("O_init"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                O_local[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                    # Load Q from gmem to smem
                                        for li, lj in T.grid(tile_x, tile_y):
                                            with T.block("Q_load"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                T.reads()
                                                T.writes()
                                                cur_L = q_indptr_val + (LH_start + i) // group_size
                                                cur_H_qo = by * group_size + (LH_start + i) % group_size
                                                if cur_L < q_indptr[b_idx + 1]:
                                                    Q_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype, rope_scaling),
                                                        q[cur_L, cur_H_qo, j]
                                                    )
                                                    # Q_smem[i, j] = q[cur_L, cur_H_qo, j]
                                                else:
                                                    Q_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                            L_kv_start: T.int32 = iterator * tile_z
                                            L_kv_base: T.int32 = kv_indptr[b_idx]
                                            for lz, ly in T.grid(tile_z, tile_y):
                                                with T.block("K_load"):
                                                    i, j = T.axis.remap("SS", [lz, ly])
                                                    T.reads()
                                                    T.writes()
                                                    cur_L = L_kv_start + i
                                                    if cur_L < kv_chunk_len[0]:
                                                        K_smem[i, j] = T.if_then_else(
                                                            rotary_mode == 1,
                                                            _rope(k, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (L_kv_base + cur_L, by, j), dtype, rope_scaling),
                                                            k[L_kv_base + cur_L, by, j]
                                                        )
                                                        # K_smem[i, j] = k[L_kv_base + cur_L, by, j]
                                                    else:
                                                        K_smem[i, j] = 0.0
                                            T.tvm_storage_sync("shared")
                                            for lz, ly in T.grid(tile_z, tile_y):
                                                with T.block("V_load"):
                                                    i, j = T.axis.remap("SS", [lz, ly])
                                                    T.reads()
                                                    T.writes()
                                                    cur_L = L_kv_start + i
                                                    if cur_L < kv_chunk_len[0]:
                                                        V_smem[i, j] = v[L_kv_base + cur_L, by, j]
                                                    else:
                                                        V_smem[i, j] = 0.0
                                            T.tvm_storage_sync("shared")

                                        # Compute S
                                            with T.block():
                                                for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                    with T.block("S_gemm"):
                                                        i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                        with T.init():
                                                            S_local[i, j] = 0.0
                                                        S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                            T.tvm_storage_sync("shared")
                                            for li, lj in T.grid(tile_x, tile_z):
                                                with T.block("S_store"):
                                                    i, j = T.axis.remap("SS", [li, lj])
                                                    S_smem[i, j] = S_local[i, j]
                                            T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                            for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                                row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                                if row < tile_x:
                                                    with T.block("update1"):
                                                        m_prev[i] = m_smem[row]
                                                        m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                        row_: T.int32 = (LH_start + row) // group_size
                                                        for j in T.serial(tile_z):
                                                            if _causal_mask(causal,
                                                                    row=row_,
                                                                    col=L_kv_start + j,
                                                                    kv_len=kv_chunk_len[0],
                                                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                                m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                        d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                            for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                                row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                        if row < tile_x:
                                                            row_: T.int32 = (LH_start + row) // group_size
                                                            if _causal_mask(causal,
                                                                    row=row_,
                                                                    col=L_kv_start + j,
                                                                    kv_len=kv_chunk_len[0],
                                                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                                S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                            else:
                                                                S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                            for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                                row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                                if row < tile_x:
                                                    with T.block("update"):
                                                        for j in T.serial(tile_z):
                                                            d_new[i] += S_smem[row, j]
                                                        m_smem[row] = m_new[i]
                                                        d_smem[row] = d_new[i]
                                                        m_prev_smem[row] = m_prev[i]
                                            T.tvm_storage_sync("shared")

                                        # Update O
                                            with T.block():
                                                for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                    with T.block("O_gemm"):
                                                        i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                        with T.init():
                                                            O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                        O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

                                    # Store O from smem to gmem
                                        for li, lj in T.grid(tile_x, tile_y):
                                            with T.block("O_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                                cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                                if cur_L < q_indptr[b_idx + 1]:
                                                    output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                        for li in T.grid(tile_x):
                                            with T.block("lse_store"):
                                                i = T.axis.remap("S", [li])
                                                cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                                cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                                if cur_L < q_indptr[b_idx + 1]:
                                                    lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                        tile_id[0] += NUM_BLKS
       
    return gpu_module





def _attention_prefill_ragged_cpu(h_kv, h_q, d, dtype, rope_scaling: Dict[str, Any]):

    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    @I.ir_module
    class cpu_module:

        @T.prim_func
        def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
            var_q: T.handle, # [total_len, h_q, d]
            var_q_indptr: T.handle, # [batch_size + 1]
            var_k: T.handle, # [total_len, h_kv, d]
            var_v: T.handle, # [total_len, h_kv, d]
            var_kv_indptr: T.handle, # [batch_size + 1]
            var_q_rope_position: T.handle, # [total_q_len]
            var_k_rope_pos_offset: T.handle, # [b]
            var_output: T.handle, # [total_len, h_q, d]
            var_lse: T.handle, # [total_len, h_q]
            causal: T.int32,
            rotary_mode: T.int32,
            rope_scale: T.float32,
            rope_theta: T.float32,
            attn_score_scaling_factor: T.float32
        ):
            batch_size = T.int32(is_size_var=True)
            qo_len = T.int32(is_size_var=True)
            kv_len = T.int32(is_size_var=True)
            q_indptr_elem_offset = T.int32(is_size_var=True)
            kv_indptr_elem_offset = T.int32(is_size_var=True)
            q_rope_position_elem_offset = T.int32(is_size_var=True)
            k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

            q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
            q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
            k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
            v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
            kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
            q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
            k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
            output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
            lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable
            
            
            # for b in T.serial(batch_size):
            #     with T.block("init"):
            #         max_val =T.alloc_buffer([1,], "int32")
            #         max_val[0] = 1000
            #     max_val[0] = T.max((kv_indptr[b + 1] - kv_indptr[b]), max_val[0])


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

                                if _causal_mask(causal,
                                                row=q_idx,
                                                col=k_idx,
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
                                            _rope(k, k_rope_pos_offset[b] + k_idx, d, rope_theta, rope_scale, (kv_indptr[b] + k_idx, h_kv_idx, d_idx), dtype, rope_scaling),
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
    return cpu_module


IR_cpu = _attention_prefill_ragged_cpu(2, 4, 256, "float32",{})
lib_cpu = tvm.build(IR_cpu, target="llvm")

IR_gpu = _attention_prefill_ragged_gpu(2, 4, 256, "float32", {})
lib_gpu = tvm.build(IR_gpu, target="cuda")


import numpy as np


def generate_batch_prefill_ragged_kv_input_data(
    batch_size = 4,        
    total_len = 500,       
    h_q = 4,               
    h_kv = 2,              
    d = 256,
    use_rope = True,
    use_causal = True                      
):
    var_q = np.random.rand(total_len, h_q, d).astype("float32")                
    var_k = np.random.rand(total_len, h_kv, d).astype("float32")              
    var_v = np.random.rand(total_len, h_kv, d).astype("float32")

    var_q_indptr = np.array([0, 100, 200, 300 ,500]).astype("int32")
    var_kv_indptr = np.array([0, 100, 200, 300 ,500]).astype("int32")

    var_q_rope_position = np.arange(total_len).astype(np.int32)              
    var_k_rope_pos_offset = np.arange(batch_size).astype(np.int32)

    var_output = np.zeros((total_len, h_q, d), dtype=np.float32)              
    var_lse = np.zeros((total_len, h_q), dtype=np.float32)

    causal = np.int32(1 if use_causal else 0)

    rotary_mode = np.int32(1 if use_rope else 0)
    rope_scale = np.float32(1.0)
    rope_theta = np.float32(10000.0)
    attn_score_scaling_factor = np.float32(1.0)                    

    return {
        "var_q": var_q,
        "var_q_indptr": var_q_indptr,
        "var_k": var_k,
        "var_v": var_v,
        "var_kv_indptr": var_kv_indptr,
        "var_q_rope_position": var_q_rope_position,
        "var_k_rope_pos_offset": var_k_rope_pos_offset,
        "var_output": var_output,
        "var_lse": var_lse,
        "causal": causal,
        "rotary_mode": rotary_mode,
        "rope_scale": rope_scale,
        "rope_theta": rope_theta,
        "attn_score_scaling_factor": attn_score_scaling_factor,
    }

inputs = generate_batch_prefill_ragged_kv_input_data()
inputs_tvm ={}
inputs_tvm_gpu = {}

for k, v in inputs.items():
    if isinstance(v, np.ndarray):
        inputs_tvm[k] = tvm.nd.array(v, device=tvm.cpu(0))
        inputs_tvm_gpu[k] = tvm.nd.array(v, device=tvm.cuda(0))
    else:
        inputs_tvm[k] = v
        inputs_tvm_gpu[k] = v

lib_cpu(*inputs_tvm.values())
lib_gpu(*inputs_tvm_gpu.values())

np.testing.assert_almost_equal(inputs_tvm["var_output"].numpy(), inputs_tvm_gpu["var_output"].numpy(), decimal=6)
np.testing.assert_almost_equal(inputs_tvm["var_lse"].numpy(), inputs_tvm_gpu["var_lse"].numpy(), decimal=5)                                          
