import tvm 
import numpy as np 
from abc import ABC

class BaseKernel(ABC):
    def __init__(self, gpu_func, cpu_func):
        self.gpu_func = gpu_func
        self.cpu_func = cpu_func
        self.args = self.gen_args()
        self.inputs_cpu = {}
        self.inputs_gpu = {}
    def gen_args(self):
        pass
    def gen_inputs(self):
        pass
    def build(type):
        pass
    def test(self,TEST):
        print("testing :",self.name)
        lib_cpu = self.build("cpu")
        lib_gpu = self.build("gpu")
        lib_cpu(*(self.inputs_cpu.values()))
        lib_gpu(*(self.inputs_gpu.values()))
        for t in TEST:
            try:
                np.testing.assert_almost_equal(self.inputs_cpu[t].numpy(), self.inputs_gpu[t].numpy(), decimal=5)
                print(t,": passed")
            except:
                print(t,": failed")
        print("Done")