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

scores, mspd_snrs, cnn_output, X, X_rebuild, dist_eval = peakeval.evaluate_peaks(peaks, pics, cal_snr=True, use_neatms=True)

dist_false, _, _ = peakeval.evaluate_noise()

plt.figure(dpi = 300, figsize = (5,3))
sns.distplot(dist_true, kde=True, bins = 40, label = 'true peaks')
sns.distplot(dist_false, kde=True, bins = 40, label = 'false peaks')
sns.distplot(dist_eval, kde=True, bins = 40, label = 'evaluating ROIs')
plt.xlabel('Euclidean distance', fontsize = 10)
plt.legend()


# remove narrow peaks
k = scores > 0

# compare AutoMS and CNN
scores_high = scores[np.logical_and(cnn_output == 1, k)]
scores_accaptable = scores[np.logical_and(cnn_output == 0, k)]
scores_noise = scores[np.logical_and(cnn_output == 2, k)]

plt.figure(dpi = 300, figsize = (5,3))
sns.distplot(scores_high, kde=True, bins = 20, label = 'high quality', color = 'red')
sns.distplot(scores_accaptable, kde=True, bins = 20, label = 'accaptable quality', color = 'blue')
sns.distplot(scores_noise, kde=True, bins = 20, label = 'noise', color = 'green')
plt.xlabel('AutoMS Score', fontsize = 10)
plt.legend()



