# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:20:12 2022

@author: jihon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from AutoMS import hpic
from AutoMS import peakeval


file = 'data/600mix_pos.mzML'
peaks, pics = hpic.hpic(file, min_intensity=500, min_snr=1, mass_inv=1, rt_inv=30)


# plot example of ROI
k = 103
pic = peaks.loc[k, 'pic_label']
pic = pics[pic]

plt.figure(dpi = 300, figsize = (4,3))
sns.scatterplot(pic[:,0], pic[:,1], hue = pic[:,2], legend = False)
plt.xlabel('RT')
plt.ylabel('m/z')


# plot distribution
X = np.load('data/X.npy')
y = np.load('data/Y.npy')

keep = []
for i in range(X.shape[0]):
    X[i,:] = X[i,:] / np.max(X[i,:])
    v = X[i,:]
    if y[i] == 1:
        if np.sum(v[23:28] > 0) == 5:
            if max(v[24], v[25]) >= np.max(v[23:28]):
                keep.append(i)
                
X_true = X[keep, :]

autoencoder = tf.keras.models.load_model('model/denoising_autoencoder.pkl')
X_rebuild = autoencoder.predict(X_true)
X_rebuild = np.reshape(X_rebuild, [-1, 50]) 
dist_true = np.array([np.linalg.norm(X_true[i] - X_rebuild[i]) for i in range(len(X_true))])

scores, mspd_snrs, cnn_output, X, X_rebuild, dist_eval = peakeval.evaluate_peaks(peaks, pics, cal_snr=True, use_cnn=True)

dist_false, _, _ = peakeval.evaluate_noise()

plt.figure(dpi = 300, figsize = (5,3))
sns.distplot(dist_true, kde=True, bins = 40, label = 'true peaks')
sns.distplot(dist_false, kde=True, bins = 40, label = 'false peaks')
sns.distplot(dist_eval, kde=True, bins = 40, label = 'evaluating ROIs')
plt.xlabel('Euclidean distance', fontsize = 10)
plt.legend()

peaks['AutoMS Score'] = scores
peaks['MSPD SNR'] = mspd_snrs
peaks['CNN Class'] = cnn_output

# remove narrow peaks
k = scores > 0
m = mspd_snrs > 0

# compare AutoMS and CNN
scores_peaks = scores[np.logical_and(cnn_output == 0, k)]
scores_noise = scores[np.logical_and(cnn_output == 1, k)]

plt.figure(dpi = 300, figsize = (5,2.5))
sns.distplot(scores_peaks, kde=True, bins = 40, label = 'CNN predicted peaks', color = 'red')
sns.distplot(scores_noise, kde=True, bins = 40, label = 'CNN predicted noises', color = 'blue')
plt.xlabel('AutoMS Score', fontsize = 10)
plt.legend()

'''
temp1 = np.where(np.logical_and(np.logical_and(scores < 1, scores > 0), cnn_output == 1))
fig, axes = plt.subplots(2, 2, dpi=300, sharex=True, sharey=True, figsize=(5, 3.2))
axes[0,0].plot(X[881,])
axes[0,0].text(27, 0.05, 'AutoMS: {} \nCNN: True'.format(np.round(scores[881], 2)), fontsize = 8)
axes[0,1].plot(X[24936,])
axes[0,1].text(27, 0.05, 'AutoMS: {} \nCNN: True'.format(np.round(scores[24936], 2)), fontsize = 8)

temp2 =  np.where(np.logical_and(scores > 2, cnn_output == 1))
axes[1,0].plot(X[8515,])
axes[1,0].text(27, 0.8, 'AutoMS: {} \nCNN: False'.format(np.round(scores[8515], 2)), fontsize = 8)
axes[1,1].plot(X[12734,])
axes[1,1].text(27, 0.8, 'AutoMS: {} \nCNN: False'.format(np.round(scores[12734], 2)), fontsize = 8)

fig.supxlabel('Scan index')
fig.supylabel('Relative intensity')
'''

# compare AutoMS and MSPD
from scipy.stats import gaussian_kde
from matplotlib import cm
from lifelines.utils import concordance_index

x_plt = scores[np.logical_and(k, m)]
y_plt = mspd_snrs[np.logical_and(k, m)]
y_plt = np.log2(y_plt)

c_index = np.round(concordance_index(x_plt, y_plt), 3)

z = np.polyfit(x_plt, y_plt, 4)
p_plt = np.polyval(z, np.arange(0, 3.5, 0.01))

xy = np.vstack([x_plt,y_plt])
z_plt= gaussian_kde(xy)(xy)

fig, ax = plt.subplots(dpi = 300, figsize=(5, 3))
ax.scatter(x_plt, y_plt, c=z_plt, s=10, cmap = 'coolwarm', alpha = 0.7)
cbar = fig.colorbar(cm.ScalarMappable(cmap = 'coolwarm'))
cbar.ax.set_ylabel('Density')
ax.plot(np.arange(0, 3.5, 0.01), p_plt, linestyle = '--', color = 'darkred')
plt.text(0, 4.0, 'c-index: {}'.format(c_index))
plt.xlabel('AutoMS Score')
plt.ylabel('Log2 SNR')
plt.show()


'''
temp1 = np.where(np.logical_and(np.logical_and(scores < 1, scores > 0), mspd_snrs > 5))
fig, axes = plt.subplots(2, 2, dpi=300, sharex=True, sharey=True, figsize=(5, 3.2))
axes[0,0].plot(X[435,])
axes[0,0].text(0, 0.05, 'AutoMS: {} \nMSPD: {}'.format(np.round(scores[881], 2), np.round(mspd_snrs[881], 2)), fontsize = 8)
axes[0,1].plot(X[1200,])
axes[0,1].text(0, 0.05, 'AutoMS: {} \nMSPD: {}'.format(np.round(scores[1200], 2), np.round(mspd_snrs[1200], 2)), fontsize = 8)

temp2 =  np.where(np.logical_and(scores > 2, np.logical_and(mspd_snrs < 2, mspd_snrs > 1)))
axes[1,0].plot(X[1366,])
axes[1,0].text(28, 0.75, 'AutoMS: {} \nMSPD: {}'.format(np.round(scores[1366], 2), np.round(mspd_snrs[1366], 2)), fontsize = 8)
axes[1,1].plot(X[15160,])
axes[1,1].text(28, 0.75, 'AutoMS: {} \nMSPD: {}'.format(np.round(scores[15160], 2), np.round(mspd_snrs[15160], 2)), fontsize = 8)

fig.supxlabel('Scan index')
fig.supylabel('Relative intensity')
'''


# compound search
import pandas as pd 
from matplotlib_venn import venn3_unweighted

peaks_refine = peaks[np.logical_and(peaks['AutoMS Score'] >= 1.30102, peaks['intensity'] > 1000)]
compounds = pd.read_csv('data/compound_list.csv', encoding = 'gbk')
AutoMS_identified = []
for i in compounds.index:
    pred_precursor = compounds.loc[i, 'ExactMass'] + 1.0078
    if np.min(np.abs(peaks_refine['mz'] - pred_precursor)) <= 0.1:
        AutoMS_identified.append(True)
    else:
        AutoMS_identified.append(False)

ms_dial = pd.read_csv('data/MS_DIAL_output.csv')
ms_dial_identified = []
for i in compounds.index:
    pred_precursor = compounds.loc[i, 'ExactMass'] + 1.0078
    if np.min(np.abs(ms_dial['Precursor m/z'] - pred_precursor)) <= 0.1:
        ms_dial_identified.append(True)
    else:
        ms_dial_identified.append(False)

xcms = pd.read_csv('data/XCMS_output.csv')
xcms_identified = []
for i in compounds.index:
    pred_precursor = compounds.loc[i, 'ExactMass'] + 1.0078
    if np.min(np.abs(xcms['mz'] - pred_precursor)) <= 0.1:
        xcms_identified.append(True)
    else:
        xcms_identified.append(False)

AutoMS_identified = list(compounds['Name'][AutoMS_identified])
ms_dial_identified = list(compounds['Name'][ms_dial_identified])
xcms_identified = list(compounds['Name'][xcms_identified])

plt.figure(dpi = 300)
venn3_unweighted([set(AutoMS_identified), set(ms_dial_identified), set(xcms_identified)], 
      ('AutoMS', 'MS-DIAL', 'XCMS'))
plt.show()


AutoMS_feat = set([str(np.round(peaks_refine['mz'][i], 1)) + '_' + str(np.round(peaks_refine['rt'][i], -1)) for i in peaks_refine.index])
ms_dial_feat = set([str(np.round(ms_dial['Precursor m/z'][i], 1)) + '_' + str(np.round(60 * ms_dial['RT (min)'][i], -1)) for i in ms_dial.index])
xcms_feat = set([str(np.round(xcms['mz'][i], 1)) + '_' + str(np.round(xcms['rt'][i], -1)) for i in xcms.index])


plt.figure(dpi = 300)
venn3_unweighted([set(AutoMS_feat), set(ms_dial_feat), set(xcms_feat)], 
      ('AutoMS', 'MS-DIAL', 'XCMS'))
plt.show()

