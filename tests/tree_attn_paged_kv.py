import tvm 
import numpy as np 
import random
from .base_kernel import BaseKernel
from .utils import get_indptr


class Tree_Attn_Paged_KV_kernel(BaseKernel):

    def __init__(self, gpu_func, cpu_func):
        self.name = "Tree_Attn_Paged_KV"
        self.gpu_func = gpu_func
        self.cpu_func = cpu_func
        self.args = {
            "N": 3,
            "QLEN": 20,
            "H_Q": 8,
            "H_KV":8,
            "D": 32,
            "PAGE_SIZE": 16,
            "MAX_PAGE": 16,
            "sliding": True,
            "dtype": "float32",
        }
        self.inputs_cpu = {}
        self.inputs_gpu = {}
    
    def test(self):
        return super().test(["output","lse"])
    
    def build(self,type):
        self.gen_inputs()
        if type == "cpu":
            IR_cpu = self.cpu_func(
                self.args["H_KV"],
                self.args["H_Q"],
                self.args["D"],
                self.args["dtype"],
                {},
                tvm.target.Target("llvm")
            )
            return tvm.build(IR_cpu,target="llvm")
        else:
            IR_gpu = self.gpu_func(
                self.args["H_KV"],
                self.args["H_Q"],
                self.args["D"],
                self.args["dtype"],
                {},
                tvm.target.Target("cuda")
            )
            return tvm.build(IR_gpu,target="cuda")

    def gen_inputs(self):
        args = self.args
        N = self.args["N"]
        QLEN = self.args["QLEN"]
        H_Q = self.args["H_Q"]
        H_KV = self.args["H_KV"]
        D = self.args["D"]
        PAGE_SIZE = self.args["PAGE_SIZE"]
        MAX_PAGE = self.args["MAX_PAGE"]
        dtype = self.args["dtype"]

        Q = np.random.rand(QLEN, H_Q, D).astype("float32")
        Q_indptr = get_indptr(QLEN,N)

        Pages = np.random.rand(MAX_PAGE, 2, H_KV, PAGE_SIZE, D).astype("float32")
        P_indptr = get_indptr(QLEN,N)
        P_value = np.random.randint(0, MAX_PAGE, size=D, dtype=np.int32)

        Slide = False,
        L_INFO = np.random.randint(0, 100, size=(N), dtype=np.int32)

        K_rope_pos_offset = np.arange(N).astype("int32")
        Q_rope_position = np.arange(QLEN).astype("int32")  


        OUTPUT = np.zeros((QLEN, H_Q, D), dtype=np.float32)          
        LSE = np.zeros((QLEN, H_Q), dtype=np.float32) 
        rotary_mode = 0
        rope_scale = random.randint(-10000,10000)/10000
        rope_theta = random.randint(-10000,10000)/10000
        attn_score_scaling_factor = random.randint(-10000,10000)/10000


        order_counts = [3, 5, 2]
        indptr = np.zeros(N + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(order_counts)
        total_len = indptr[-1]
        tree_order = np.zeros((total_len, 2), dtype=np.int32)
        for tree_idx in range(N):
            start = indptr[tree_idx]
            end = indptr[tree_idx + 1]
            for i, pos in enumerate(range(start, end)):
                tree_order[pos, 0] = i          
                tree_order[pos, 1] = tree_idx   

        inputs =  {
            "_0": 0,
            "var_q": Q,
            "var_q_indptr":  Q_indptr,
            "var_pages": Pages,
            "var_page_indptr": P_indptr,
            "var_page_values": P_value, # [nnz_pages]
            "var_length_info": L_INFO, # [b] when sliding window = False, or otherwise [3, b]
            "var_k_rope_pos_offset": K_rope_pos_offset, # [b]
            "var_q_rope_position": Q_rope_position, # [total_len]
            "output": OUTPUT, # [total_len, h_q, d]
            "lse": LSE, # [total_len, h_q]
            "rotary_mode": 0,
            "rope_scale": rope_scale,
            "rope_theta": rope_theta,
            "attn_score_scaling_factor": attn_score_scaling_factor,
            "tree_indptr": indptr,
            "tree_handle": tree_order
        }
        self.inputs_cpu = {}
        self.inputs_gpu = {}

        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                self.inputs_cpu[k] = tvm.nd.array(v, device=tvm.cpu(0))
                self.inputs_gpu[k] = tvm.nd.array(v, device=tvm.cuda(0))
            else:
                self.inputs_cpu[k] = v
                self.inputs_gpu[k] = v