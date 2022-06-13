# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:11:16 2022

@author: DELL
"""


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from AutoMS.mspd_original import peaks_detection

# load data
x = np.load('data/X.npy')
y = np.load('data/Y.npy')

for i in range(x.shape[0]):
    x[i,:] = x[i,:] / np.max(x[i,:])
    
sums = np.sum(x, axis = 1)
keep = np.where(sums > 0)[0]
x = x[keep, :]
y = y[keep]


pos = np.where(y == 1)[0]
x_true = x[pos,:]

neg = np.where(y == 0)[0]
x_false = x[neg,:]

false = []
for i in range(x_false.shape[0]): 
    pks, sigs, snrs_ = peaks_detection(x_false[i,:], np.arange(1, 30), 0)
    criterion_1 = np.sum(x_false[i,:] == 0) < 5
    if not criterion_1:
        continue
    snrs_.append(0)
    criterion_2 = np.max(snrs_) <= 3
    if criterion_2:
        false.append(i)
    if len(false) == 99474:
        break
false = np.array(false)
x_false = x_false[false,:]

x = np.vstack((x_true, x_false))
y = np.array([1] * len(x_true) + [0] * len(x_false))
y = np.vstack((y, 1-y)).T

class CNN:
    def __init__(self, X, Y):
        X = np.expand_dims(X, -1)
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
        
        inp = Input(shape=(X.shape[1:]))
        hid = inp
        
        # layer 1
        hid = Conv1D(32, kernel_size=2, activation='relu')(hid)
        hid = MaxPooling1D(pool_size=2)(hid)
        
        # layer 2        
        hid = Conv1D(16, kernel_size=2, activation='relu')(hid)
        hid = MaxPooling1D(pool_size=2)(hid)
        
        # layer 3        
        hid = Conv1D(16, kernel_size=2, activation='relu')(hid)
        hid = MaxPooling1D(pool_size=2)(hid)
        
        # layer dense
        hid = Flatten()(hid)
        hid = Dense(32, activation="relu")(hid)
        
        # output layer
        prd = Dense(2, activation="softmax")(hid)
        opt = optimizers.Adam(lr=0.001)
        model = Model(inp, prd)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        self.model = model
    
    def train(self, epochs=5):
        self.model.fit(self.X_tr, self.Y_tr, validation_split= 0.1, epochs=epochs)
    
    def test(self):
        Y_pred = np.round(self.model.predict(self.X_ts))
        f1 = f1_score(self.Y_ts[:,0], Y_pred[:,0])
        precision = precision_score(self.Y_ts[:,0], Y_pred[:,0])
        recall = recall_score(self.Y_ts[:,0], Y_pred[:,0])
        accuracy = accuracy_score(self.Y_ts[:,0], Y_pred[:,0])
        return accuracy, precision, recall, f1
    
    def save(self, path):
        self.model.save(path)


cnn = CNN(x, y)
cnn.train(epochs = 10)
cnn.test()
cnn.save('model/cnn.pkl')

