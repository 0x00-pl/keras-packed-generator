import numpy as np


class packer:
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


class unpacker:
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


    def close(self):
        self.f.close()


def main():
    pk = packer('test.data')
    for i in range(10000):
        arr = np.arange(128*64).reshape(128, 64, 1).astype('int16')
        pk.add_ndarray(arr, np.arange(400).astype('float'))

    pk.export_index('test.info')
    pk.close()

    up = unpacker('test.data', np.load('test.info.npy'))
    for i in range(10000):
        arr = up.get_data_from_idx(i)


if __name__ == '__main__':
    main()
