import tvm 
import numpy as np 
import random
from .base_kernel import BaseKernel
from .utils import get_indptr

class Merge_State_Inplace_kernel(BaseKernel):

    def __init__(self, gpu_func, cpu_func):
        self.name = "Merge_State_Inplace"
        self.gpu_func = gpu_func
        self.cpu_func = cpu_func
        self.args = {
            "N": 256,
            "H": 32,
            "D": 32,
            "dtype": "float32",
        }
        self.inputs_cpu = {}
        self.inputs_gpu = {}
    
    def test(self):
        return super().test(["V"])
    
    def build(self,type):
        self.gen_inputs()
        if type == "cpu":
            IR_cpu = self.cpu_func(
                self.args["H"],
                self.args["D"],
                self.args["dtype"],
                tvm.target.Target("llvm")
            )
            return tvm.build(IR_cpu,target="llvm")
        else:
            IR_gpu = self.gpu_func(
                self.args["H"],
                self.args["D"],
                self.args["dtype"],
                tvm.target.Target("cuda")
            )
            return tvm.build(IR_gpu,target="cuda")

    def gen_inputs(self):
        N = self.args["N"]
        H = self.args["H"]
        D = self.args["D"]
        dtype = self.args["dtype"]


        V = np.random.rand(N, H, D).astype(np.float32)
        S = np.random.rand(N, H).astype(np.float32)
        V_other = np.random.rand(N, H, D).astype(np.float32)
        S_other = np.random.rand(N, H).astype(np.float32)

        inputs =  {
            "V": V,
            "S": S,
            "VO": V_other,
            "SO": S_other,
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