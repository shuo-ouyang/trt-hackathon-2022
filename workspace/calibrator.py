import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

calib_data_path = './calibration'
cache_file = './int8.cache'


def load_calib_data():
    calib_data = []
    for i in range(160):
        data = np.load(f"./calibration/calib-{i}.npz")
        calib_data.append(data['images'])
    return np.array(calib_data).astype(np.float32).reshape(5120, 3, 224, 224)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_data, cache_file, batch_size=32) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = load_calib_data()
        self.current_index = 0
        print('calib data shape: ', self.data.shape)

        self.device_inputs = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        current_batch = int(self.current_index // self.batch_size)
        if self.current_index % 32 == 0:
            print("Calibrating batch {:}, containing {:} images".format(
                current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size]
        cuda.memcpy_htod(self.device_inputs, batch)
        self.current_index += self.batch_size
        return [self.device_inputs]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
