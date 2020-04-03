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

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def show_single_image(img_arr):
    plt.imshow(img_arr,cmap="binary")
    plt.show()

def show_imgs(n_rows ,n_cols ,x_data ,y_data ,class_names):
    assert len(x_data) == len(y_data)
    assert n_cols*n_rows<len(x_data)
    plt.figure(figsize = (n_cols*1.4 , n_rows*1.6))#创建自定义图像
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols*row + col
            plt.subplot(n_cols,n_rows,index+1)#函数的index是从1开始，但是定义的index从0开始
            plt.imshow(x_data[index],cmap="binary",interpolation="nearest")#interpolation是缩放图片插值方法
            plt.axis('off')#关闭坐标系
            plt.title(class_names[y_data[index]])
    plt.show()

def Sequential_model():
    #relu:y = max{0,x}
    #softmax:将变量变成概率分布.x=[x1,x2,x3]
    #    y = [e^x1/sum,e^x2/sum,e^x3/sum],sum=e^x1 + e^x2 + e^x3
    #y是一个index值，想要求得损失函数要将y通过one_hot变成一个向量 

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    model.add(keras.layers.Dense(300,activation="relu"))
    model.add(keras.layers.Dense(100,activation="relu"))
    model.add(keras.layers.Dense(10,activation="softmax"))

    #另外一种网络写法
    #model = keras.Sequential([
    #    keras.layers.Flatten(input_shape=(28, 28)),
    #    keras.layers.Dense(128, activation='relu'),
    #    keras.layers.Dense(10, activation='softmax')
    #])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))
    plot_learning_curves(history)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

Sequential_model()