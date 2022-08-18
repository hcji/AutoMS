# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:21:29 2022

@author: DELL
"""


import pandas as pd

compounds = pd.read_csv('data/compound_list.csv', encoding = 'gbk')


# AutoMS
from AutoMS.automs import AutoMS

file = 'data/600mix_pos.mzML'
peaks = AutoMS(file, min_intensity = 500)
peaks = peaks.sort_values('score', ascending=False, ignore_index=True)

# XCMS
xcms = pd.read_csv('data/XCMS_output.csv')
xcms = xcms.sort_values('sn', ascending=False, ignore_index=True)

# MS-DIAL
ms_dial = pd.read_csv('data/MS_DIAL_output.csv')
ms_dial = ms_dial.sort_values('S/N', ascending=False, ignore_index=True)

# Peakonly
peakonly = pd.read_csv('data/peakonly_output.csv')


import numpy as np
from tqdm import tqdm

def count_true_positive(data, colname = 'Precursor m/z'):
    identified = []
    count = 0
    for i in compounds.index:
        pred_precursor = compounds.loc[i, 'ExactMass'] + 1.0078
        if np.min(np.abs(data[colname] - pred_precursor)) <= 0.1:
            identified.append(True)
            count += 1
        else:
            identified.append(False)
    return count, identified


x = np.arange(1, 6600, 300)
y_automs, y_xcms, y_msdial = [], [], []
for n in tqdm(x):
    y_automs_, _ = count_true_positive(peaks.loc[:n,:], 'mz')
    y_xcms_, _ = count_true_positive(xcms.loc[:n,:], 'mz')
    y_msdial_, _ = count_true_positive(ms_dial.loc[:n,:], 'Precursor m/z')
    y_automs.append(y_automs_)
    y_xcms.append(y_xcms_)
    y_msdial.append(y_msdial_)





