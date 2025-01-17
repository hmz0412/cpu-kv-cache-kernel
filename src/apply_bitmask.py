import tvm
from tvm import IRModule
from tvm.script import tir as T


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