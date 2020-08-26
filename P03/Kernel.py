import numpy as np
class Kernel:
    def __init__(self, *args):
        self.kernels = []
        for kernel in args:
            kernel = np.array(kernel, np.float32)
            kernel = np.transpose(kernel)
            self.kernels.append(kernel)
    def __str__(self):
        count = 1
        outStr = ""
        for kernel in self.kernels:
            outStr += "Kernel {}: \n{}\n".format(count, kernel)
        return outStr