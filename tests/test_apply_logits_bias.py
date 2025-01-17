import tvm
from tvm import IRModule
from tvm.script import tir as T
import numpy as np

from mlc_llm.support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)


def _get_apply_logit_bias_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    if max_num_threads_per_block < tx:
        tx = max_num_threads_per_block
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_logit_bias_inplace(
        var_logits: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_logit_bias: T.handle,
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        # seq_ids
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        logit_bias = T.match_buffer(var_logit_bias, (num_token,), "float32")

        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vp = T.axis.spatial(num_token, p0 * tx + p1)
                    T.where(p0 * tx + p1 < num_token)
                    logits[pos2seq_id[vp], token_ids[vp]] += logit_bias[vp]
    return _apply_logit_bias_inplace



def _get_apply_logit_bias_inplace_cpu():
    tx = 1024
    @T.prim_func
    def _apply_logit_bias_inplace(
        var_logits: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_logit_bias: T.handle,
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        # seq_ids
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        logit_bias = T.match_buffer(var_logit_bias, (num_token,), "float32")

        # for p0 in T.parallel(0, (num_token + tx - 1) // tx):
        #     for p1 in T.vectorized(0, tx):
        #         with T.block("block"):
        #             vp = T.axis.spatial(num_token, p0 * tx + p1)
        #             T.where(p0 * tx + p1 < num_token)
        #             logits[pos2seq_id[vp], token_ids[vp]] += logit_bias[vp]

        
        for token in T.serial(num_token):
            logits[pos2seq_id[token], token_ids[token]] += logit_bias[token]

    return _apply_logit_bias_inplace

IR_cpu = _get_apply_logit_bias_inplace_cpu()
lib_cpu = tvm.build(IR_cpu, target="llvm")

IR_gpu = _get_apply_logit_bias_inplace(tvm.target.Target("cuda"))
lib_gpu = tvm.build(IR_gpu, target="cuda")


import numpy as np

def generate_test_data(batch_size, vocab_size, num_token):
    """
    Generates a larger test dataset for `_apply_logit_bias_inplace` function
    where no single point is updated multiple times.

    Parameters:
    - batch_size (int): The number of sequences in the batch.
    - vocab_size (int): The size of the vocabulary.
    - num_token (int): The total number of tokens to process.

    Returns:
    - dict: A dictionary containing the generated test data.
    """
    assert batch_size * vocab_size >= num_token, \
        "Ensure that the number of unique points (batch_size * vocab_size) >= num_token."
    
    # Generate unique (pos2seq_id, token_ids) pairs
    all_possible_pairs = [(i, j) for i in range(batch_size) for j in range(vocab_size)]
    selected_pairs = np.random.choice(len(all_possible_pairs), num_token, replace=False)
    pos2seq_id, token_ids = zip(*[all_possible_pairs[idx] for idx in selected_pairs])
    
    pos2seq_id = np.array(pos2seq_id, dtype="int32")
    token_ids = np.array(token_ids, dtype="int32")
    
    # Generate logits initialized to zeros
    logits = np.zeros((batch_size, vocab_size), dtype="float32")
    
    # Generate random logit_bias values
    logit_bias = np.random.rand(num_token).astype("float32")
    
    return {
        "logits": logits,
        "pos2seq_id": pos2seq_id,
        "token_ids": token_ids,
        "logit_bias": logit_bias,
    }




test_data = generate_test_data(batch_size=10, vocab_size=10, num_token=100)

test_data_cpu = {}
test_data_gpu = {}

for k, v in test_data.items():
    test_data_gpu[k] = tvm.nd.array(v, device=tvm.cuda(0))
    test_data_cpu[k] = tvm.nd.array(v, device=tvm.cpu(0))


lib_gpu(*test_data_gpu.values())
lib_cpu(*test_data_cpu.values())

np.testing.assert_almost_equal(test_data_cpu["logits"].numpy(), test_data_gpu["logits"].numpy(), decimal=7)
