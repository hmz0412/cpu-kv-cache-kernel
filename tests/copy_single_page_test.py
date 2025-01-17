import tvm 
import numpy as np 
import random
from .base_kernel import BaseKernel
from .utils import get_indptr

class Copy_Single_Page_kernel(BaseKernel):

    def __init__(self, gpu_func, cpu_func):
        self.name = "Copy_Single_Page"
        self.gpu_func = gpu_func
        self.cpu_func = cpu_func
        self.args = {
            "N": 12,
            "H": 12,
            "D": 10,
            "PZ": 256,
            "NP": 1024,
            "dtype": "float32",
        }
        self.inputs_cpu = {}
        self.inputs_gpu = {}
    
    def test(self):
        return super().test(["Pages"])
    
    def build(self,type):
        self.gen_inputs()
        if type == "cpu":
            IR_cpu = self.cpu_func(
                self.args["H"],
                self.args["PZ"],
                self.args["D"],
                self.args["dtype"],
                tvm.target.Target("llvm")
            )
            return tvm.build(IR_cpu,target="llvm")
        else:
            IR_gpu = self.gpu_func(
                self.args["H"],
                self.args["PZ"],
                self.args["D"],
                self.args["dtype"],
                tvm.target.Target("cuda")
            )
            return tvm.build(IR_gpu,target="cuda")

    def gen_inputs(self):
        args = self.args
        N = self.args["N"]
        H = self.args["H"]
        D = self.args["D"]
        PZ = self.args["PZ"]
        NP= self.args["NP"]
        dtype = self.args["dtype"]


        Pages = np.random.rand(NP,2,H,PZ,D).astype(np.float32)

        inputs =  {
            "Pages": Pages,
            "srcid":  123,
            "tgtid": 456,
            "length_size": 128,
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