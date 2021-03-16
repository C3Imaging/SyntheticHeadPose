import numpy as np


def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


def get_ypr_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        pose = np.asarray(f.readline().strip().split(','), dtype='float32')
    return pose


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
