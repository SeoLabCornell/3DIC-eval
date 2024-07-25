import numpy as np

def get_sram_energy(buf_size):
    return 0.1321453125 * np.sqrt(buf_size)


def get_sram_area(buf_size):
    return (buf_size / 512) * 0.75  # mm2


def get_mac_energy():
    # energy = 0.435  # 0.435 pJ/MAC Google TPU (ISCA 2017) Jouppi et al.
    energy = 0.2296 # 0.2224 pJ/MAC TSMC 28nm synthesis estimation * 2
    return energy


def get_3d_array_size(dim_2d):
    return np.round(dim_2d / np.sqrt(2))