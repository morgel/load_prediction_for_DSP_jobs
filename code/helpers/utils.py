import GPUtil
from threading import Thread
import time
import numpy as np

class GPUMonitor(Thread):
    def __init__(self, delay, index):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        
        self.delay = delay # Time between calls to GPUtil
        self.util_values = []
        
        GPUs = GPUtil.getGPUs()
        self.gpu = GPUs[index]
        
        self.start()
        
    @property
    def gpu_util(self):
        return np.array(self.util_values).mean()

    def run(self):
        while not self.stopped:
            self.util_values.append(self.gpu.memoryUtil)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        