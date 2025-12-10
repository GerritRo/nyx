import numpy as np

def plane_parallel(Z):
    # Clip airmass to be a maximum of 40 at 90 deg zenith
    return  1 / np.maximum(np.cos(Z), 0.025)

def kasten_young_1989(Z):
    return 1/(np.cos(Z) + 0.50572*(96.07995-np.rad2deg(Z))**(-1.6364))