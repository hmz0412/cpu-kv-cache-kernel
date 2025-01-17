import tvm
from tvm import IRModule
from tvm.script import tir as T


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