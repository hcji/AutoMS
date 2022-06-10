# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:20:12 2022

@author: jihon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from peakeval import hpic
from peakeval import peakeval


file = 'data/600mix_pos.mzML'
peaks, pics = hpic.hpic(file, min_intensity=500, min_snr=1, mass_inv=1)


# plot example of ROI
k = 103
pic = peaks.loc[k, 'pic_label']
pic = pics[pic]

plt.figure(dpi = 300, figsize = (4,3))
sns.scatterplot(pic[:,0], pic[:,1], hue = pic[:,2], legend = False)
plt.ylim(44.04, 44.06)
plt.xlabel('RT')
plt.ylabel('m/z')


# plot distribution
X = np.load('data/X.npy')
y = np.load('data/Y.npy')

keep = []
for i in range(X.shape[0]):
    X[i,:] = X[i,:] / np.max(X[i,:])
    n = np.sum(X[i,23:28] > 0) == 5
    keep.append(n)

true = np.where(np.logical_and(y == 1, keep))[0]
X_true = X[true, :]

autoencoder = tf.keras.models.load_model('model/denoising_autoencoder.pkl')
X_rebuild = autoencoder.predict(X_true)
X_rebuild = np.reshape(X_rebuild, [-1, 50]) 
dist_true = np.array([np.linalg.norm(X_true[i] - X_rebuild[i]) for i in range(len(X_true))])

_, _, _, _, dist_eval = peakeval.evaluate_peaks(peaks, pics)

dist_false, _, _ = peakeval.evaluate_noise()

plt.figure(dpi = 300, figsize = (5,3))
sns.distplot(dist_true, kde=True, bins = 40, label = 'true peaks')
sns.distplot(dist_false, kde=True, bins = 40, label = 'false peaks')
sns.distplot(dist_eval, kde=True, bins = 40, label = 'evaluating ROIs')
plt.xlabel('Euclidean distance', fontsize = 10)
plt.legend()

