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

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

#x = (x - u)/std   u是均值，std是方差，使数据符合均值是0，方差是1的正态分布
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.fit_transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.fit_transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)


def Sequential_model():
    #relu:y = max{0,x}
    #softmax:将变量变成概率分布.x=[x1,x2,x3]
    #    y = [e^x1/sum,e^x2/sum,e^x3/sum],sum=e^x1 + e^x2 + e^x3
    #y是一个index值，想要求得损失函数要将y通过one_hot变成一个向量 

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    for _ in range(20) :
        model.add(keras.layers.Dense(100,activation='relu'))
        model.add(keras.layers.BatchNormalization())#批归一化
        """
        激活函数放在批归一化之后的方法
        model.add(keras.layers.Dense(100))不做激活
        model.add(keras.layers.Batch.Normalization())批归一化
        model.add(keras.layers.Activation('relu'))
        """
    model.add(keras.layers.Dense(10,activation='softmax'))

    #另外一种网络写法
    #model = keras.Sequential([
    #    keras.layers.Flatten(input_shape=(28, 28)),
    #    keras.layers.Dense(128, activation='relu'),
    #    keras.layers.Dense(10, activation='softmax')
    #])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    logdir = './dnn-callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
    ]
    history = model.fit(x_train_scaled,y_train,epochs=10,validation_data=(x_valid_scaled,y_valid),callbacks=callbacks)
    plot_learning_curves(history)
    model.evaluate(x_test_scaled,y_test)
    model.summary()
"""
accuracy增长缓慢的原因：
1:参数众多，训练不充分
2.梯度消失->链式法则->复合函数f(g(x))
批归一化缓解梯度消失
"""
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

Sequential_model()