import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#强制使用CPU
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import sys
import time
import tensorflow as tf
import pprint

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

housing = fetch_california_housing()

#分割训练集和测试集
x_train_all,x_test,y_train_all,y_test = train_test_split(
    housing.data,housing.target,random_state=7
)
#分割验证集和训练集
x_train,x_valid,y_train,y_valid = train_test_split(
    x_train_all,y_train_all,random_state=11
)

#print(x_train.shape,y_train.shape)
#print(x_valid.shape,y_valid.shape)
#print(x_test.shape,y_test.shape)

#归一化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#learning rate:[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
#W = W + grad * learning_rate
leraning_rates = [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
histories = []
for lr in leraning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30,activation='relu',input_shape=x_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss='mean_squared_error',optimizer=optimizer)#sgd梯度下降
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5,min_delta=1e-4)
    ]
    history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=100,callbacks=callbacks)
    histories.append(history)
for history in histories:
    plot_learning_curves(history)
plot_learning_curves(history)
