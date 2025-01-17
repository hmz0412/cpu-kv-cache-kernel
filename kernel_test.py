import src as k
import tests as t

def test_compact_kv_copy():
    kernel = t.Compact_KV_Copy_kernel(gpu_func = k._compact_kv_copy, cpu_func = k._compact_kv_copy_cpu)
    kernel.test()

def test_copy_single_page():
    kernel = t.Copy_Single_Page_kernel(gpu_func = k._copy_single_page, cpu_func= k._copy_single_page_cpu)
    kernel.test()

def test_merge_state_inplace():
    kernel = t.Merge_State_Inplace_kernel(gpu_func= k._merge_state_inplace, cpu_func = k._merge_state_inplace_cpu)
    kernel.test()

def test_attention_prefill():
    kernel = t.Attention_Prefill_kernel(gpu_func=k._attention_prefill, cpu_func= k._attention_prefill_cpu)
    kernel.test()

def test_tree_attn_paged_kv():
    kernel = t.Tree_Attn_Paged_KV_kernel(gpu_func=k.tree_attn_with_paged_kv_cache, cpu_func=k._tree_attn_paged_kv_cpu)
    kernel.test()

test_merge_state_inplace()
test_copy_single_page()
test_compact_kv_copy()
test_attention_prefill()
test_tree_attn_paged_kv()
