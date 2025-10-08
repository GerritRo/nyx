import numpy as np

def plane_parallel(Z):
    return 1/np.cos(Z)

def kasten_young_1989(Z):
    return 1/(np.cos(Z) + 0.50572*(96.07995-np.rad2deg(Z))**(-1.6364))