#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:39:03 2018

@author: parth
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data_raw = pd.read_csv('./creditcard.csv')
x_train = np.array(data_raw.iloc[:,0:29])
y_train = np.array(data_raw.iloc[:,30])


# architect a simple binary feedforward nerual network with relu activations

m = x_train.shape[0]
n = x_train.shape[1]

x = tf.placeholder(tf.float32,shape=x_train.shape, name="input")
y = tf.placeholder(tf.float32, shape=[m,None],name="output")

W1 = tf.Variable(tf.random_normal((n,10)) * np.sqrt(2/n),name="W1")
b1 = tf.Variable(tf.random_normal((1,10)), name="b1")
Z1 = tf.matmul(x,W1) + b1
A1 = tf.nn.relu(Z1)