# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# load data
x = np.load('data/X.npy')
y = np.load('data/Y.npy')

sums = np.sum(x, axis = 1)
keep = np.where(np.logical_and(y == 1, sums > 0))[0]
x = x[keep, :]
y = y[keep]

for i in range(x.shape[0]):
    x[i,:] = x[i,:] / np.max(x[i,:])


x_train, x_test, _, _ = train_test_split(x, y, test_size=0.1)
del(y)

# preprocessing
length = x_train.shape[1]
x_train = np.reshape(x_train, [-1, length, 1])
x_test = np.reshape(x_test, [-1, length, 1])


# random noise layer
input_shape = (length, 1)

class RandomNoise(keras.layers.Layer):
    def call(self, inputs):
        noise = K.random_normal(shape=tf.shape(inputs), stddev = 0.2)
        outputs = K.in_train_phase(inputs + noise, inputs)
        return (outputs) / tf.keras.backend.max(outputs)


# encoder
inputs = keras.layers.Input(shape=input_shape,name='encoder_input')

x = inputs
x = RandomNoise()(x)
# x = keras.layers.GaussianNoise(0.1)(x, training = True)
# x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv1D(8, 5, activation='relu')(x)
x = keras.layers.Conv1D(4, 5, activation='relu')(x)
x = keras.layers.Flatten()(x)
latent = keras.layers.Dense(64, name='latent_vector')(x)
encoder = keras.Model(inputs,latent,name='encoder')

# decoder
latent_inputs = keras.layers.Input(shape=(64,),name='decoder_input')
x = keras.layers.Dense(32, activation='relu')(latent_inputs)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dense(length, activation='linear')(x)
outputs = keras.layers.Reshape((length, 1))(x)
decoder = keras.Model(latent_inputs,outputs,name='decoder')

# model
autoencoder = keras.Model(inputs, decoder(encoder(inputs)),name='autoencoder')
autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=10)
autoencoder.save('model/denoising_autoencoder.pkl')

# predict
x_decoded = autoencoder.predict(x_test)
x_test = np.reshape(x_test, [-1, length])
x_decoded = np.reshape(x_decoded, [-1, length])


plt.figure(dpi = 300, figsize = (12,4))
plt.subplot(241)
plt.plot(x_test[1,:])
plt.subplot(242)
plt.plot(x_test[2,:])
plt.subplot(243)
plt.plot(x_test[3,:])
plt.subplot(244)
plt.plot(x_test[4,:])
plt.subplot(245)
plt.plot(x_decoded[1,:])
plt.subplot(246)
plt.plot(x_decoded[2,:])
plt.subplot(247)
plt.plot(x_decoded[3,:])
plt.subplot(248)
plt.plot(x_decoded[4,:])
