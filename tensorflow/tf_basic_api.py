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

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(t)
print(t[:,1:])
print(t[...,1])

#ops
print(t+10)
print(tf.square(t))
print(t @ tf.transpose(t))

#numpy conversion
print(t.numpy)
print(np.square(t))
np_t = np.array([[1.,2.,3.],[4.,5.,6.]])
print(tf.constant(np_t))

#Scalars
t = tf.constant(2.718)
print(t.numpy)
print(t.shape)

#Strings
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t,unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t,"UTF8"))

#string array
t = tf.constant(["cafe","coffee","咖啡"])
print(tf.strings.length(t,unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t,"UTF-8")
print(r)

#ragged tensor 不完整的Tensor，各维度长度不相同
r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
#index op
print(r)
print(r[1])
print(r[1:2])

#ops on ragged tensor
r2 = tf.ragged.constant([[51,52],[],[71]])
r3 = tf.ragged.constant([[13,14],[15],[],[42,43]])
print(tf.concat([r,r3],axis=0))
