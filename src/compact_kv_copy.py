from typing import Any, Dict, Tuple

import tvm
from tvm.script import tir as T
from tvm.target import Target


def _compact_kv_copy_cpu(num_heads, head_dim, dtype, target: Target):
    tx = 8

    @T.prim_func
    def compact_kv_copy_cpu(
        var_pages: T.handle,
        var_copy_length_indptr: T.handle,
        var_copy_src_dst_pos: T.handle,
        batch_size: T.int32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, 16, head_dim), dtype)
        copy_length_indptr = T.match_buffer(
            var_copy_length_indptr,
            (batch_size + 1,),
            "int32",
            elem_offset=copy_length_indptr_elem_offset,
        )
        copy_src_dst_pos = T.match_buffer(
            var_copy_src_dst_pos,
            (2, total_copy_length),
            "int32",
            elem_offset=copy_src_dst_pos_elem_offset,
        )

        # with T.block("root"):
        #     for b in T.serial(batch_size):
        #         for h in T.serial(num_heads):
        #             for d in T.serial(head_dim):
        #                 for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
        #                     src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
        #                     dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
        #                     pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
        #                     pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]


        with T.block("root"):
            for bhd_o in T.serial((batch_size * num_heads * head_dim + tx -1) // tx ) :
                for bhd_i in T.serial(tx):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                            pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    return compact_kv_copy_cpu