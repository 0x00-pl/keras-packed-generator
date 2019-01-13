import random

import numpy as np


class Packer:
    def __init__(self, output_file):
        self.of = open(output_file, 'wb')
        self.info = []

    def add_ndarray(self, input_arr, output_arr):
        ipos = self.of.tell()
        idata = input_arr.tobytes()
        self.of.write(idata)

        opos = self.of.tell()
        odata = output_arr.tobytes()
        self.of.write(odata)

        self.info.append([
            ipos, len(idata), input_arr.dtype, input_arr.shape,
            opos, len(odata), output_arr.dtype, output_arr.shape
        ])

    def export_index(self, filename):
        np.save(filename, self.info)

    def close(self):
        self.of.close()


class Unpacker:
    def __init__(self, input_file, info):
        self.f = open(input_file, 'rb')
        self.info = info

    def get_data_from_idx(self, idx):
        info = self.info[idx]
        return self.get_data(info)

    def get_data(self, info):
        istart, ilen, idtype, ishape, ostart, olen, odtype, oshape = info
        self.f.seek(istart)
        idata = self.f.read(ilen)
        iflatten = np.frombuffer(idata, idtype)
        input = iflatten.reshape(ishape)

        self.f.seek(ostart)
        odata = self.f.read(olen)
        oflatten = np.frombuffer(odata, odtype)
        output = oflatten.reshape(oshape)
        return input, output

    def get_data_part_from_idx(self, idx, i_part_start_count, i_part_shape, isize=None):
        info = self.info[idx]
        return self.get_data_part(info, i_part_start_count, i_part_shape, isize)

    def get_data_part(self, info, i_part_start_count, i_part_shape, isize=None):
        istart, ilen, idtype, ishape, ostart, olen, odtype, oshape = info
        isize = isize or np.dtype(idtype).itemsize
        i_part_len_count = np.product(i_part_shape)
        self.f.seek(istart + isize * i_part_start_count)
        idata = self.f.read(isize * i_part_len_count)
        iflatten = np.frombuffer(idata, idtype)
        input = iflatten.reshape(i_part_shape)

        self.f.seek(ostart)
        odata = self.f.read(olen)
        oflatten = np.frombuffer(odata, odtype)
        output = oflatten.reshape(oshape)
        return input, output

    def close(self):
        self.f.close()


def main():
    pk = Packer('test.data')
    for i in range(10000):
        arr = np.arange(128 * 64).reshape(128, 64, 1).astype('int16')
        pk.add_ndarray(arr, np.arange(400).astype('float'))

    pk.export_index('test.info')
    pk.close()

    up = Unpacker('test.data', np.load('test.info.npy'))
    for i in range(10000):
        arr = up.get_data_from_idx(i)

    up = Unpacker('test.data', np.load('test.info.npy'))
    for i in range(10000):
        arr = up.get_data_part_from_idx(i, random.randrange(0, 128 * 64) - 10, [10])


if __name__ == '__main__':
    main()
