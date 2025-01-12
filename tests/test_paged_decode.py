# pylint: disable=too-many-statements,too-many-lines,too-many-arguments,invalid-name
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


def get_max_num_threads_per_block(target: Target) -> int:
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = target.max_num_threads
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)


def check_thread_limits(target: Target, bdx: int, bdy: int, bdz: int, gdz: int):
    """
    Check whether max num threads exceeded given a target.

    Parameters
    ----------
    bdx: threadIdx.x
    bdy: threadIdx.y
    bdz: threadIdx.z
    gdz: blockIdx.z
    """
    max_num_threads_per_block = get_max_num_threads_per_block(target)

    assert (
        bdx * bdy * bdz <= max_num_threads_per_block
    ), f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{max_num_threads_per_block}"

    if str(target.kind) == "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's threadIdx.z cannot exceed 64, but got bdz={bdz}"
        assert gdz == 1, f"webgpu's blockIdx.z should be 1, but got gdz={gdz}"

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


def _attention_decode_gpu(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    sliding_window: bool,
    rope_scaling: Dict[str, Any],
    target: Target,
):
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    THREAD_LIMIT = 512
    TILE_SIZE_PER_BDX = 2 #每個block處理tile的size 2 tokens
    if target.kind.name == "opencl" and "android" in str(target.host):
        THREAD_LIMIT = 256 if H_kv < 8 else 512
        TILE_SIZE_PER_BDX = 1
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    thread_limit = min(max_num_threads_per_block, THREAD_LIMIT)
    

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = min(max(8 // qkv_dtype_bytes, D // 32), 4)
    bdx = D // VEC_SIZE # 全部維度/一次可處理的 -> 用多少thread
    bdy = GROUP_SIZE
    while bdx * bdy > thread_limit and bdy > 1:
        bdy //= 2
    gdz = GROUP_SIZE // bdy #用gdz補足原本 group size的部分
    threads_per_CTA = max(thread_limit, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy) #看是不是有多出來的thread 可以往z方向延伸
    tile_size_per_bdx = TILE_SIZE_PER_BDX if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=bdz, gdz=1)

    global_symbol = "batch_decode_paged_kv"
    if sliding_window:
        global_symbol += "_sliding_window"

    # pylint: disable=line-too-long,too-many-branches
    # fmt: off
    @T.prim_func
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

        for bx in T.thread_binding(B, thread="blockIdx.x"):
            for fused_by_bz in T.thread_binding(H_kv * gdz, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        for tz in T.thread_binding(bdz, thread="threadIdx.z"):
                            with T.block("attn"):
                                Q_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                K_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                V_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                O_allreduce = T.alloc_buffer((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = T.alloc_buffer((bdz, bdy, 2), "float32", scope="shared")
                                S_reduce_local = T.alloc_buffer((1,), "float32", scope="local")
                                t0 = T.alloc_buffer((1,), "float32", scope="local")

                                S_local = T.alloc_buffer((bdy * tile_size_per_bdx), "float32", scope="local")
                                QK_local = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")
                                V_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                m_prev = T.alloc_buffer((1,), "float32", scope="local")
                                d_prev = T.alloc_buffer((1,), "float32", scope="local")
                                other_m = T.alloc_buffer((1,), "float32", scope="local")
                                other_d = T.alloc_buffer((1,), "float32", scope="local")
                                exp_mprev = T.alloc_buffer((1,), "float32", scope="local")
                                exp_otherm = T.alloc_buffer((1,), "float32", scope="local")
                                other_o = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")
                                st_m = T.alloc_buffer((1,), "float32", scope="local")
                                st_d = T.alloc_buffer((1,), "float32", scope="local")
                                O_local = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")

                                by: T.int32 = fused_by_bz % H_kv #第幾個kv head
                                bz: T.int32 = fused_by_bz // H_kv #gdz
                                batch_idx: T.int32 = bx
                                cur_page_indptr_begin: T.int32 = page_table_indptr[batch_idx]
                                cur_page_indptr_end: T.int32 = page_table_indptr[batch_idx + 1]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, batch_idx, length_info, sliding_window),
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(Q, q_rope_position[batch_idx], head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec), qkv_dtype, rope_scaling),
                                        Q[bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec] #by * GROUP_SIZE -> 第幾個kv group。bz * bdy bz是gdz  bz*bdy -> group +ty group上第幾個 head
                                    )

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx  # type: ignore share mem 
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx  # type: ignore, global mem
                                    # load KV from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        with T.block("KV_load"):
                                            T.reads()
                                            T.writes()
                                            row_g: T.int32(is_size_var=True) = tile_start_g + j  # type: ignore 在global裡面一直往下找
                                            if row_g < kv_chunk_len[0]:
                                                seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_g, batch_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, 16)  # type: ignore
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[batch_idx] + row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec), qkv_dtype, rope_scaling),
                                                        pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec]
                                                    )
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                            else:
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        # compute S = Q * K * sm_scale
                                        for vec in T.vectorized(VEC_SIZE):
                                            QK_local[vec] = T.cast(Q_local[vec], "float32") * T.cast(K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec], "float32") * attn_score_scaling_factor * sm_scale
                                        S_reduce_local[0] = 0
                                        for vec in T.unroll(VEC_SIZE):
                                            S_reduce_local[0] += QK_local[vec]
                                        
                                        #不同塊把它合起來
                                        with T.block("block_cross_thread"):
                                            T.reads(S_reduce_local[0])
                                            T.writes(t0[0])
                                            T.attr(
                                                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                                "reduce_scope",
                                                T.reinterpret("handle", T.uint64(0)),
                                            )
                                            T.tvm_thread_allreduce(T.uint32(1), S_reduce_local[0], True, t0[0], tx, dtype="handle")

                                        S_local[j] = -5e4
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j < kv_chunk_len[0]:
                                            S_local[j] = t0[0]
                                        # update st_m
                                        st_m[0] = T.max(st_m[0], S_local[j])

                                    # update st_d, st_O
                                    o_scale: T.float32 = T.exp2(m_prev[0] - st_m[0])
                                    st_d[0] *= o_scale
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        S_local[j] = T.exp2(S_local[j] - st_m[0]) #softmax 分子
                                        st_d[0] += S_local[j] #softmax 分母
                                    for j in T.vectorized(VEC_SIZE):
                                        O_local[j] *= o_scale

                                    # load V from shared memory to local memory
                                    # compute O
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        for vec in T.vectorized(VEC_SIZE):
                                            V_local[vec] = V_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] += T.cast(V_local[vec], "float32") * S_local[j] # V * S

                                if bdz > 1:
                                    # allreduce over bdz
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_allreduce[tz, ty, tx * VEC_SIZE + vec] = O_local[vec]
                                    md_allreduce[tz, ty, 0] = st_m[0]
                                    md_allreduce[tz, ty, 1] = st_d[0]
                                    T.tvm_storage_sync("shared")

                                    st_m[0] = -5e4
                                    st_d[0] = 1.0
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_local[vec] = 0.0

                                    for j in T.serial(bdz):
                                        m_prev[0] = st_m[0]
                                        d_prev[0] = st_d[0]
                                        other_m[0] = md_allreduce[j, ty, 0]
                                        other_d[0] = md_allreduce[j, ty, 1]
                                        for vec in T.vectorized(VEC_SIZE):
                                            other_o[vec] = O_allreduce[j, ty, tx * VEC_SIZE + vec]
                                        st_m[0] = T.max(st_m[0], other_m[0])
                                        st_d[0] = d_prev[0] * T.exp2(m_prev[0] - st_m[0]) + other_d[0] * T.exp2(other_m[0] - st_m[0])
                                        exp_mprev[0] = T.exp2(m_prev[0] - st_m[0])
                                        exp_otherm[0] = T.exp2(other_m[0] - st_m[0])
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] = O_local[vec] * exp_mprev[0] + other_o[vec] * exp_otherm[0]

                                # normalize O
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] /= st_d[0]

                                # store O to global memory
                                for vec in T.vectorized(VEC_SIZE):
                                    output[batch_idx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec] = O_local[vec]

                                # store lse to global memory
                                lse[batch_idx, by * GROUP_SIZE + bz * bdy + ty] = st_m[0] + T.log2(st_d[0])
    # fmt: on
    # pylint: enable=line-too-long,too-many-branches
    return batch_decode_paged_kv

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

IR_cpu = _attention_decode_cpu(2, 8, 1024, "float32", True, {})
lib_cpu = tvm.build(IR_cpu, target="llvm")

IR_gpu = _attention_decode_gpu(2, 8, 1024, "float32", True, {}, Target("cuda"))
lib_gpu = tvm.build(IR_gpu, target="cuda")                  


import numpy as np

def generate_inputs_for_batch_decode_paged_kv(
    B=2,              
    H_qo=8,           
    H_kv=2,           
    D=1024,              
    max_num_pages=3,  
    nnz_pages=3,      
    sliding_window=True
):

   
    Q = np.random.randn(B, H_qo, D).astype("float32")

    pages = np.random.randn(max_num_pages, 2, H_kv, 16, D).astype("float32")
    page_table_indptr = np.array([0, 2, 3], dtype="int32")
    page_table_values = np.array([0, 1, 2], dtype="int32")
   
    if not sliding_window:
        var_length_info = np.array([7, 5], dtype="int32")
    else:
        # shape (3, B) = (3,2)
        # (0,b) => last_page_len
        # (1,b) => sliding window initial offset
        # (2,b) => attn sink length
        # ex: batch 0: [7, 2, 9], batch 1: [5, 3, 8]
        var_length_info = np.array([[7, 5],
                                    [2, 3],
                                    [9, 8]], dtype="int32")

    
    k_rope_pos_offset = np.arange(B).astype(np.int32)
    q_rope_position = np.arange(B).astype(np.int32)

    output = np.zeros((B, H_qo, D), dtype="float32")
    lse = np.zeros((B, H_qo), dtype="float32")

    rotary_mode = np.int32(1)                 
    rope_scale = np.float32(1.0)             
    rope_theta = np.float32(10000.0)         
    attn_score_scaling_factor = np.float32(1.0)

    return {
        "Q": Q,
        "pages": pages,
        "page_table_indptr": page_table_indptr,
        "page_table_values": page_table_values,
        "var_length_info": var_length_info,
        "k_rope_pos_offset": k_rope_pos_offset,
        "q_rope_position": q_rope_position,
        "output": output,
        "lse": lse,
        "rotary_mode": rotary_mode,
        "rope_scale": rope_scale,
        "rope_theta": rope_theta,
        "attn_score_scaling_factor": attn_score_scaling_factor,
    }


inputs = generate_inputs_for_batch_decode_paged_kv()
inputs_tvm = {}
inputs_tvm_gpu = {}
for k, v in inputs.items():
    if isinstance(v, np.ndarray):
        inputs_tvm[k] = tvm.nd.array(v, device=tvm.cpu(0))
        inputs_tvm_gpu[k] = tvm.nd.array(v, device=tvm.cuda(0))
    else:
        inputs_tvm[k] = v
        inputs_tvm_gpu[k] = v



lib_cpu(0, *inputs_tvm.values())
lib_gpu(0, *inputs_tvm_gpu.values())

np.testing.assert_almost_equal(inputs_tvm["output"].numpy(), inputs_tvm_gpu["output"].numpy(), decimal=5)
np.testing.assert_almost_equal(inputs_tvm["lse"].numpy(), inputs_tvm_gpu["lse"].numpy(), decimal=5)

                

    