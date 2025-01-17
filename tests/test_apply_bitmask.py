import tvm
from tvm import IRModule
from tvm.script import tir as T
import numpy as np

from mlc_llm.support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)

def _get_apply_bitmask_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    if max_num_threads_per_block < tx:
        tx = max_num_threads_per_block
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_bitmask_inplace(
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_bitmask: T.handle,
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        bitmask = T.match_buffer(var_bitmask, (batch_size, (vocab_size + 31) // 32), "int32")

        for fused_s_v_0 in T.thread_binding(0, (num_seq * vocab_size + tx - 1) // tx, "blockIdx.x"):
            for fused_s_v_1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vs = T.axis.spatial(num_seq, (fused_s_v_0 * tx + fused_s_v_1) // vocab_size)
                    vv = T.axis.spatial(vocab_size, (fused_s_v_0 * tx + fused_s_v_1) % vocab_size)
                    T.where(fused_s_v_0 * tx + fused_s_v_1 < num_seq * vocab_size)
                    logits[seq_ids[vs], vv] = T.if_then_else(
                        (bitmask[seq_ids[vs], vv // 32] >> (vv % 32)) & 1 == 1,
                        logits[seq_ids[vs], vv],
                        T.min_value("float32"),
                    )

    return _apply_bitmask_inplace


def _get_apply_bitmask_inplace_cpu():

    @T.prim_func
    def _apply_bitmask_inplace(
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_bitmask: T.handle,
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        bitmask = T.match_buffer(var_bitmask, (batch_size, (vocab_size + 31) // 32), "int32")
        
        for token in T.serial(num_seq * vocab_size):
            with T.block("block"):
                vs = T.axis.spatial(num_seq, (token) // vocab_size)
                vv = T.axis.spatial(vocab_size, (token) % vocab_size)

                logits[seq_ids[vs], vv] = T.if_then_else(
                    (bitmask[seq_ids[vs], vv // 32] >> (vv % 32) ) & 1 == 1,
                    logits[seq_ids[vs], vv],
                    T.min_value("float32"),
                )

    return _apply_bitmask_inplace

def generate_inputs_fixed(batch_size, vocab_size, num_seq):
    # Generate logits as random float32 values
    logits = np.random.rand(batch_size, vocab_size).astype(np.float32)
    
    # Generate seq_ids as random integers within batch_size range
    seq_ids = np.array([0, 1, 2]).astype(np.int32)
    
    # Generate bitmask: Randomly decide 0 or 1 for each bit
    # Each integer in bitmask can represent 32 vocabulary items
    bitmask_width = (vocab_size + 31) // 32  # Number of 32-bit integers needed
    bitmask = np.zeros((batch_size, bitmask_width), dtype=np.int32)
    
    for i in range(batch_size):
        # Randomly assign valid bits in the bitmask
        for j in range(vocab_size):
            if np.random.rand() > 0.5:
                bitmask[i, j // 32] |= (1 << (j % 32))
    
    return {
        "logits": logits,
        "seq_ids": seq_ids,
        "bitmask": bitmask
    }

IR_cpu = _get_apply_bitmask_inplace_cpu()
IR_gpu = _get_apply_bitmask_inplace(tvm.target.Target("cuda"))

lib_cpu = tvm.build(IR_cpu, target="llvm")
lib_gpu = tvm.build(IR_gpu, target="cuda")

# Example usage
inputs = generate_inputs_fixed(batch_size=4, vocab_size=100, num_seq=3)

inputs_cpu = {}
inputs_gpu = {}

for k, v in inputs.items():
    inputs_cpu[k] = tvm.nd.array(v, tvm.cpu(0))
    inputs_gpu[k] = tvm.nd.array(v, tvm.cuda(0))

lib_cpu(*inputs_cpu.values())
lib_gpu(*inputs_gpu.values())

#print(np.abs(inputs_cpu["logits"].numpy() - inputs_gpu["logits"].numpy()))

#np.testing.assert_almost_equal(inputs_cpu["logits"].numpy(), inputs_gpu["logits"].numpy(), decimal=4)

np.testing.assert_allclose(
    inputs_cpu["logits"].numpy(),
    inputs_gpu["logits"].numpy(),
    rtol=1e-5,  
    atol=1e-7   
)
