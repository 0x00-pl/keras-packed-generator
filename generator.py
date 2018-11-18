import random

import numpy as np

import pack_npy


class g:
    def __init__(self, filename, infofilename):
        self.up = pack_npy.unpacker(filename, np.load(infofilename))

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
                    iarr, oarr = self.up.get_data(info)
                    inputs.append(iarr)
                    outputs.append(oarr)
                yield np.array(inputs), np.array(outputs)

    def __len__(self):
        return len(self.up.info)
