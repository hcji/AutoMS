# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:02:51 2022

@author: jihon
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fitter import Fitter
from scipy.stats import t
from sklearn.preprocessing import MinMaxScaler
from AutoMS.mspd_original import peaks_detection


def evaluate_noise():
    X = np.load('data/X.npy')
    y = np.load('data/Y.npy')
    
    false = np.where(y == 0)[0]
    X = X[false,:]
    np.random.shuffle(X)
    
    false = []
    for i in range(X.shape[0]): 
        pks, sigs, snrs_ = peaks_detection(X[i,:], np.arange(1, 30), 3)
        criterion_1 = np.sum(X[i,:] == 0) < 5
        if not criterion_1:
            continue
        X[i,:] = X[i,:] / np.max(X[i,:])
        snrs_.append(0)
        criterion_2 = np.max(snrs_) <= 3
        if criterion_2:
            false.append(i)
        if len(false) == 20000:
            break

    false = np.array(false)
    X_noise = X[false, :]
    
    autoencoder = tf.keras.models.load_model('model/denoising_autoencoder.pkl')
    X_rebuild = autoencoder.predict(X_noise)
    X_rebuild = np.reshape(X_rebuild, [-1, 50])

    distance = np.array([np.linalg.norm(X_noise[i] - X_rebuild[i]) for i in range(len(X_noise))])
    
    f = Fitter(distance, distributions=['norm', 't', 'f'])
    f.fit()
    f.get_best()
    
    '''
    plt.figure(dpi = 300)
    sns.kdeplot(distance)
    '''
    return distance, X_noise, X_rebuild


def evaluate_peaks(peaks, pics, length=14, params=(8.5101, 1.6113, 0.1950), min_width = 6, cal_snr=False, use_cnn=False):
    traces, snrs = [], []
    exclude = []

    for i in tqdm(peaks.index):
        rt = peaks.loc[i, 'rt']
        pic = peaks.loc[i, 'pic_label']
        pic = pics[pic]
        
        x = np.linspace(rt - length, rt + length, 50)
        x0, y0 = pic[:,0], pic[:,2]
        y = np.interp(x, x0, y0)
        y = y / np.max(y)
        traces.append(y)
        
        if max(y[24], y[25]) < np.max(y[int(25-min_width/2):int(25+min_width/2)]):
            exclude.append(i)
        elif np.min(y[int(25-min_width/2):int(25+min_width/2)]) < 0.3:
            exclude.append(i)
        else:
            pass
        
        if cal_snr:
            pks, sigs, snrs_ = peaks_detection(y, np.arange(1, 30), 0)
            if (len(snrs_) == 0) or (np.min(np.abs(np.array(pks) - 25)) > 3):
                snr = 0
            else:
                wh = np.argmin(np.abs(np.array(pks) - 25))
                snr = snrs_[wh]
            snrs.append(snr)
            
    snrs = np.array(snrs)
    traces = np.array(traces)
    exclude = np.array(exclude)
    
    X = traces
    autoencoder = tf.keras.models.load_model('model/denoising_autoencoder.pkl')
    X_rebuild = autoencoder.predict(X)
    X_rebuild = np.reshape(X_rebuild, [-1, 50])
    
    distance = np.array([np.linalg.norm(X[i] - X_rebuild[i]) for i in range(len(X))])
    scores = t.pdf(distance, params[0], loc = params[1], scale = params[2])
    scores = -np.log10(scores)
    scores[exclude] = 0
    
    if use_cnn:
        cnn = tf.keras.models.load_model('model/cnn.pkl')
        classes = cnn.predict(X)
        cnn_output = np.argmax(classes, axis = 1)
    else:
        cnn_output = []
        
    
    '''
    k = 3233
    print(scores[k])
    y = X[k,:]
    y2 = X_rebuild[k,:]
    
    plt.figure(dpi = 300, figsize = (4.5,3))
    plt.plot(y, lw = 3, label = 'original')
    plt.fill_between(np.arange(50), y, color = 'lightblue', alpha = 0.7)

    plt.plot(y2, lw = 3, color = 'red', label = 'reconstructed')
    plt.fill_between(np.arange(50), y2, color = 'lightpink', alpha = 0.7)
    plt.legend(loc = 'upper left')
    plt.xlabel('scan index')
    plt.ylabel('relative intensity')
    
    plt.figure(dpi = 300)
    y1 = y + np.random.normal(0, 0.1, size = y.shape)
    plt.plot(y1, lw = 3)
    plt.fill_between(np.arange(50), y1, color = 'lightblue')
    '''
    
    return scores, snrs, cnn_output, X, X_rebuild, distance
