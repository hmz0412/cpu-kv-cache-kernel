import tvm
from tvm import IRModule
from tvm.script import tir as T
import numpy as np

from mlc_llm.support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)

def _get_apply_penalty_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    if max_num_threads_per_block < tx:
        tx = max_num_threads_per_block
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_penalty_inplace(  # pylint: disable=too-many-arguments,too-many-locals
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_token_cnt: T.handle,
        var_penalties: T.handle,
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        token_cnt = T.match_buffer(var_token_cnt, (num_token,), "int32")
        penalties = T.match_buffer(var_penalties, (num_seq, 3), "float32")

        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vp = T.axis.spatial(num_token, p0 * tx + p1)
                    T.where(p0 * tx + p1 < num_token)
                    # Penalties: (presence_penalty, frequency_penalty, repetition_penalty)
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                        penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                    )
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < 0,
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        * penalties[pos2seq_id[vp], 2],
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        / penalties[pos2seq_id[vp], 2],
                    )

    return _apply_penalty_inplace

def _get_apply_penalty_inplace_cpu():

    @T.prim_func
    def _apply_penalty_inplace(  # pylint: disable=too-many-arguments,too-many-locals
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_token_cnt: T.handle,
        var_penalties: T.handle,
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        token_cnt = T.match_buffer(var_token_cnt, (num_token,), "int32")
        penalties = T.match_buffer(var_penalties, (num_seq, 3), "float32")

        for token in T.serial(num_token):
            with T.block("block"):
                vp = T.axis.spatial(num_token, token)
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                    penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                )
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < 0,
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                    * penalties[pos2seq_id[vp], 2],
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                    / penalties[pos2seq_id[vp], 2],
                )

    return _apply_penalty_inplace


def generate_apply_penalty_inputs(batch_size, vocab_size, num_token, num_seq):
    """
    Generates input data for the `_apply_penalty_inplace` function, ensuring unique (seq_id, token_id)
    pairs so no single point is penalized more than once.

    Args:
        batch_size (int): Size of the batch (number of sequences).
        vocab_size (int): Size of the vocabulary.
        num_token (int): Total number of tokens in the batch.
        num_seq (int): Number of sequences in the batch.

    Returns:
        dict: A dictionary containing all the input arrays with correct properties.
    """
    assert batch_size >= num_seq, "Number of sequences (num_seq) cannot exceed batch size."
    assert batch_size * vocab_size >= num_token, "Ensure unique (seq_id, token_id) pairs."

    # Generate unique (seq_id, token_id) pairs
    all_possible_pairs = [(i, j) for i in range(num_seq) for j in range(vocab_size)]
    selected_pairs = np.random.choice(len(all_possible_pairs), num_token, replace=False)
    pos2seq_id, token_ids = zip(*[all_possible_pairs[idx] for idx in selected_pairs])
    
    pos2seq_id = np.array(pos2seq_id, dtype="int32")
    token_ids = np.array(token_ids, dtype="int32")
    
    # Generate logits initialized to random values
    logits = np.random.uniform(-1, 1, size=(batch_size, vocab_size)).astype("float32")
    
    # Generate sequence IDs (each sequence maps to one row in logits)
    seq_ids = np.arange(num_seq).astype("int32")
    
    # Generate random token counts (to simulate frequency)
    token_cnt = np.random.randint(1, 10, size=num_token).astype("int32")
    
    # Generate penalties (presence_penalty, frequency_penalty, repetition_penalty)
    penalties = np.random.uniform(0.1, 2.0, size=(num_seq, 3)).astype("float32")
    
    # Pack all inputs into a dictionary
    inputs = {
        "logits": logits,
        "seq_ids": seq_ids,
        "pos2seq_id": pos2seq_id,
        "token_ids": token_ids,
        "token_cnt": token_cnt,
        "penalties": penalties
    }
    
    return inputs



test_inputs = generate_apply_penalty_inputs(batch_size=10, vocab_size=10, num_token=100, num_seq=10)

inputs_cpu = {}
inputs_gpu = {}

for k, v in test_inputs.items():
    inputs_cpu[k] = tvm.nd.array(v, device=tvm.cpu(0))
    inputs_gpu[k] = tvm.nd.array(v, device=tvm.cuda(0))

IR_gpu = _get_apply_penalty_inplace(tvm.target.Target("cuda"))
IR_cpu = _get_apply_penalty_inplace_cpu()

lib_gpu = tvm.build(IR_gpu, target="cuda")
lib_cpu = tvm.build(IR_cpu, target="llvm")

lib_gpu(*inputs_gpu.values())
lib_cpu(*inputs_cpu.values())

np.testing.assert_almost_equal(inputs_cpu["logits"].numpy(), inputs_gpu["logits"].numpy(), decimal=6)