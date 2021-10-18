#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('classification.csv')
df


# <h3>Feature Selection</h3>

# In[3]:


x_data = df[['cap-shape','cap-color','odor','stalk-color-above-ring','stalk-color-above-ring','veil-color','ring-number','habitat']]
x_data


# In[4]:


y_data = df[['class']]
y_data


# In[5]:


x_data['cap-shape'] = x_data['cap-shape'].map( {'b': 1, 'c': 2, 'x': 3, 'f': 4, 'k': 5, 's': 6} )
x_data.odor = x_data.odor.map( {'a': 1, 'l': 2, 'c': 3, 'y': 4, 'f': 5, 'm': 6, 'n': 7, 'p': 8, 's': 9} )
x_data.habitat = x_data.habitat.map( {'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6, 'd': 7} )
x_data


# In[6]:


y_data = OneHotEncoder().fit_transform(y_data).toarray()
y_data


# <h3>Feature Extraction</h3>

# In[7]:


x_data = x_data.to_numpy()
x_data


# In[8]:


x_data = StandardScaler().fit_transform(x_data)
x_data


# In[9]:


x_data = PCA(n_components=4).fit_transform(x_data)
x_data


# <h3>Architecture</h3>

# In[10]:


# initialization
layer = {
    'input': 4,
    'hidden': 8,
    'output': 2
}

weight = {
    'th': tf.Variable(tf.random_normal([layer['input'],layer['hidden']])),
    'to': tf.Variable(tf.random_normal([layer['hidden'],layer['output']]))
}

bias = {
    'th': tf.Variable(tf.random_normal([layer['hidden']])),
    'to': tf.Variable(tf.random_normal([layer['output']]))
}


# In[11]:


# split database
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(1/3))

x = tf.placeholder(tf.float32,[None, layer['input']])
target = tf.placeholder(tf.float32,[None, layer['output']])


# In[12]:


# forward passing
def forward_pass():
    #to hidden layer
    wx_b1 = tf.matmul(x, weight['th']) + bias['th']
    y1 = tf.nn.softmax(wx_b1)
    
    #to output layer
    wx_b2 = tf.matmul(y1, weight['to']) + bias['to']
    y2 = tf.nn.softmax(wx_b2)
    
    return y2


# In[13]:


y = forward_pass()


# In[14]:


# variables for training and testing
EPOCH = 2500
ALPHA = 0.09

# MSE(Mean Squared Error)
error = tf.reduce_mean(0.5 * (target - y) ** 2)

# runs Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(ALPHA)

# minimizes the error values
train = optimizer.minimize(error)


# In[15]:


saver = tf.train.Saver()


# In[16]:


val_errors = math.inf
accuracy = None


# In[17]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCH):
        
        #training
        sess.run(train, feed_dict = {x: x_train, target: y_train})
        # counting error of each epoch
        current_error = sess.run(error, feed_dict={x: x_train, target: y_train})
        true_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(target, axis=1))
        accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))
        accuracy = sess.run(accuracy, feed_dict={x: x_test, target: y_test})
        
        if i % 25 == 0:
            print(f'EPOCH {i} || ERROR: {current_error: .2f}')
        
        if i == 125:
            val_error = sess.run(error, feed_dict={x: x_val, target: y_val})
            saver.save(sess, './bpnn-model.ckpt')
        
        if i % 125 == 0:
            val_error_temp = sess.run(error, feed_dict={x: x_val, target: y_val})
            if val_error_temp < val_errors:
                val_errors = val_error_temp
                saver.save(sess, './bpnn-best-error-model.ckpt')


# <h3>Evaluation</h3>

# In[18]:


accuracy = accuracy * 100
# accuracy = ((len(y_pred))/(len(y_test)))*100
print(f'Accuracy: {accuracy: .1f}%')

