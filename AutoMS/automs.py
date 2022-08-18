# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:04:55 2022

@author: DELL
"""

import numpy as np
from AutoMS import hpic
from AutoMS import peakeval


def AutoMS(file, min_intensity=500, mass_inv=1, rt_inv=30, length=14, 
           params=(8.5101, 1.6113, 0.1950), min_width = 6):
    
    peaks, pics = hpic.hpic(file, min_intensity=min_intensity, min_snr=1, mass_inv=1, rt_inv=30)
    scores, mspd_snrs, _, _, _ = peakeval.evaluate_peaks(peaks, pics, length=length, 
                                                         params=params, min_width = min_width, 
                                                         cal_snr=False)
    scores = np.array(scores)
    scores[scores < 0] = 0
    peaks['score'] = scores
    return peaks



if __name__ == '__main__':
    
    file = 'data/600mix_pos.mzML'
    peaks = AutoMS(file, min_intensity = 50000)
    