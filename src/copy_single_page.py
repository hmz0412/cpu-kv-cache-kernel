from tvm.script import tir as T
from tvm.target import Target


def _copy_single_page_cpu(num_heads, page_size, head_dim, dtype, target: Target):
    tx = 1 
    @T.prim_func
    def copy_single_page_cpu(
        var_pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)

        # for b in T.thread_binding(
        #     (copy_length * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
        # ):
        for b in T.serial((copy_length * num_heads * head_dim + tx - 1) // tx):
            for t in T.serial(tx):
                with T.block("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(
                        num_heads,
                        T.Cast("int32", (b * tx + t) // (copy_length * head_dim)),
                    )
                    vp = T.axis.spatial(
                        copy_length,
                        (b * tx + t) % (copy_length * head_dim) // head_dim,
                    )
                    vd = T.axis.spatial(
                        head_dim,
                        T.Cast(
                            "int32",
                            (b * tx + t) % head_dim,
                        ),
                    )
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page_cpu