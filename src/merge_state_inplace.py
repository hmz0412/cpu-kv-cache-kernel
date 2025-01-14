from typing import Any, Dict, Tuple
import tvm
from tvm.script import tir as T
from tvm.target import Target


def _var_cpu(dtype):
    return T.alloc_buffer((1,), dtype)

def _merge_state_inplace_cpu(num_heads, head_dim, v_dtype, target: Target):

    @T.prim_func
    def merge_state_inplace_cpu(
        v: T.handle,
        s: T.handle,
        v_other: T.handle,
        s_other: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        N = T.int32(is_size_var=True)
        H = T.int32(is_size_var=True)
        D = T.int32(is_size_var=True)

        V = T.match_buffer(v, (N, H, D), v_dtype)
        S = T.match_buffer(s, (N, H), "float32")
        V_other = T.match_buffer(v_other, (N, H, D), v_dtype)
        S_other = T.match_buffer(s_other, (N, H), "float32")


        for n in T.serial(N):
             for h in T.serial(H):
                with T.block("merge"): 
                    s_val = _var_cpu("float32")
                    s_other_val = _var_cpu("float32")
                    s_max = _var_cpu("float32")
                    scale = _var_cpu("float32")
                    other_scale = _var_cpu("float32")

                    s_val[0] = S[n,h]
                    s_other_val[0] = S_other[n,h]
                    s_max[0] = T.max(s_val[0], s_other_val[0])
                    s_val[0] = T.exp2(s_val[0] - s_max[0])
                    s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                    scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                    other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                    
                    for d in T.serial(D): 
                        V[n,h,d] =  V[n,h,d] * scale[0] + V_other[n,h,d] * other_scale[0]
                    
                    S[n,h] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    return merge_state_inplace_cpu
