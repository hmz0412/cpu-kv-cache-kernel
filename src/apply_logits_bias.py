import tvm
from tvm import IRModule
from tvm.script import tir as T




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
