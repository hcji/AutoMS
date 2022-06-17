# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:25:03 2022

@author: DELL
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from AutoMS import hpic
from AutoMS import peakeval

path = 'E:/Data/MTBLS413'
files = os.listdir(path)

output = {}
for f in tqdm(files):
    file = path + '/' + f
    peaks, pics = hpic.hpic(file, min_intensity=500, min_snr=1, mass_inv=1, rt_inv=30)
    scores, _, _, _, _, _ = peakeval.evaluate_peaks(peaks, pics, cal_snr=False, use_cnn=False)
    peaks['scores'] = scores
    output[f] = peaks
    

rtmzs = []
for i, peaks in output.items():
    for j in peaks.index:
        if peaks['scores'][j] <= 0:
            continue
        mz = np.round(peaks['mz'][j], 2)
        rt = np.round(peaks['rt'][j], -1)
        rtmz = str(rt) + '_' + str(mz)
        if rtmz not in rtmzs:
            rtmzs.append(rtmz)

intensities = np.full((len(rtmzs), len(output)), np.nan)
scores = np.full((len(rtmzs), len(output)), np.nan)
for i, peaks in output.items():
    for j in peaks.index:
        mz = np.round(peaks['mz'][j], 2)
        rt = np.round(peaks['rt'][j], -1)
        intensity = peaks['intensity'][j]
        score = peaks['scores'][j]
        rtmz = str(rt) + '_' + str(mz)
        if rtmz not in rtmzs:
            continue
        else:
            k = rtmzs.index(rtmz)
            m = files.index(i)
            intensities[k, m] = np.nanmax([intensities[k, m], intensity])
            scores[k, m] = score

keep = np.sum(~np.isnan(intensities), axis = 1) > 22 * 0.7
intensities = intensities[keep,:]
scores = scores[keep,:]

rsds = np.nanstd(intensities, axis = 1) / np.nanmean(intensities, axis = 1) * 100
mean_scores = np.nanmean(scores, axis = 1)

bins = []
for s in mean_scores:
    if s >= 3:
        bins.append('score >= 3')
    elif s >= 2:
        bins.append('2 <= score < 3')
    elif s >= 1:
        bins.append('1 <= score < 2')
    else:
        bins.append('score < 1')

plt.figure(dpi = 300, figsize = (5,3))
sns.violinplot(x = bins, y = rsds)
plt.ylim(0, 50)
plt.xticks(rotation = 15)
plt.ylabel('RSD')

print( np.median(rsds [np.array(bins) == 'score < 1'] ) )

