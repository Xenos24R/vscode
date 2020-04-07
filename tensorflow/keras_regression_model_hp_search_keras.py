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
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


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

#f(x) = 1/(x*loa(b/a))  a <= x <= b
param_distribution = {
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate":reciprocal(1e-4,1e-2),
}

def build_model(hidden_layers = 1,layer_size = 30,learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size,activation='relu',input_shape=x_train.shape[1:]))
    
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size,activation='relu'))
    
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    return model

sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
callbacks = [
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-4)
]

random_search_cv = RandomizedSearchCV(sklearn_model,param_distribution,n_iter=10,n_jobs=1)
random_search_cv.fit(x_train_scaled,y_train,epochs=100,validation_data=(x_valid_scaled,y_valid),callbacks=callbacks)

print(random_search_cv.best_index_)
print(random_search_cv.best_params_)
print(random_search_cv.best_estimator_)



