import random

import numpy as np

import pack_npy


class Generator:
    def __init__(self, filename, info_filename):
        self.up = pack_npy.Unpacker(filename, np.load(info_filename))

    def __call__(self, batch_size=32, *args, **kwargs):
        while True:
            random.shuffle(self.up.info)
            idx = 0
            while idx < len(self.up.info):
                info_list = self.up.info[idx: idx + batch_size]
                idx += batch_size
                inputs = []
                outputs = []
                for info in info_list:
                    i_arr, o_arr = self.up.get_data(info)
                    inputs.append(i_arr)
                    outputs.append(o_arr)
                yield np.array(inputs), np.array(outputs)

    def __len__(self):
        return len(self.up.info)
