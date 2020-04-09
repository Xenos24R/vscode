import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#强制使用CPU
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import sys
import time
import tensorflow as tf

from tensorflow import keras

#tf.function and autograph
def scaled_elu(z,scale=1.0,alpha=1.0):
    #z>=0 ? scale*z : scale*alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z,0.0)
    return scale * tf.where(is_positive,z,alpha*tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3.,-2.5])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3.,-2.5])))