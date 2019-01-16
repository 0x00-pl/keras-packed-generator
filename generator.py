import random

import numpy as np

import pack_npy


class FullData:
    def __init__(self, filename, info_filename):
        self.up = pack_npy.Unpacker(filename, np.load(info_filename))

    def __call__(self, batch_size=32, *args, **kwargs):
        while True:
            random.shuffle(self.up.info_list)
            idx = 0
            while idx < len(self.up.info_list):
                info_list = self.up.info_list[idx: idx + batch_size]
                idx += batch_size
                inputs = []
                outputs = []
                for info in info_list:
                    i_arr, o_arr = self.up.get_data(info)
                    inputs.append(i_arr)
                    outputs.append(o_arr)
                yield np.array(inputs), np.array(outputs)

    def __len__(self):
        return len(self.up.info_list)


class InputPartData:
    def __init__(self, filename, info_filename):
        self.up = pack_npy.Unpacker(filename, np.load(info_filename))

    def __call__(self, num_items_each_i_part, i_part_shape, batch_size=32, *args, **kwargs):
        while True:
            inputs = []
            outputs = []
            info_list = random.choices(self.up.info_list, k=batch_size)
            for info in info_list:
                istart, ilen, idtype, ishape, ostart, olen, odtype, oshape = info
                isize = np.dtype(idtype).itemsize
                part_start = random.randrange(0, int(ilen / isize) - num_items_each_i_part)
                i_arr, o_arr = self.up.get_data_part(info, part_start, i_part_shape, isize)
                inputs.append(i_arr)
                outputs.append(o_arr)
            yield np.array(inputs), np.array(outputs)

    def __len__(self):
        return len(self.up.info_list)


def generator_middleware(generator, inputs_middleware=None, targets_middleware=None, sample_weights_middleware=None):
    for datas in generator:
        ret = list(datas)
        if inputs_middleware:
            ret[0] = inputs_middleware(datas[0])
        if targets_middleware:
            ret[1] = targets_middleware(datas[1])
        if sample_weights_middleware:
            ret[2] = sample_weights_middleware(datas[2])
        yield ret
