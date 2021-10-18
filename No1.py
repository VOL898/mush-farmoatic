#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf


# <h3>SOM Architecture</h3>

# In[2]:


def get_distance(x, weight):
    return tf.sqrt(tf.reduce_mean((x - weight) ** 2, axis = 1))

class SOM:
    # Constructor
    def __init__(self, width, height, n_features, learning_rate):
        self.width = width
        self.height = height
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        self.weight = tf.Variable(
            tf.random.normal(
                [width * height, n_features]
            ),tf.float32
        )
        
        self.input = tf.placeholder(
            tf.float32,
            [n_features]
        )
        
        self.location = []
        for y in range(height):
            for x in range(width):
                self.location.append(tf.cast([y,x], tf.float32))
                
        self.cluster = [ [] for i in range(height) ]
        
        self.bmu = self.get_bmu()
        self.update = self.update_neighbor()
        
    def get_bmu(self):
        distance = get_distance(self.input, self.weight)
        
        bmu_index = tf.argmin(distance)
        bmu_location = tf.cast([tf.div(bmu_index, self.width), tf.mod(bmu_index, self.width)], tf.float32)
        
        return bmu_location
    
    def update_neighbor(self):
        
        # Find Rate
        distance = get_distance(self.bmu, self.location)
        sigma = tf.cast(tf.maximum(self.width, self.height) / 2, tf.float32)    
        neighbor_strength = tf.exp(-(distance ** 2) / (2 * sigma ** 2))
        rate = neighbor_strength * self.learning_rate
        
        stacked_rate = []
        for i in range(self.width * self.height):
            stacked_rate.append(tf.tile([rate[i]], [self.n_features]))
        
        # Update
        delta = ( stacked_rate * (self.input - self.weight))
        new_weight = self.weight + delta
            
        return tf.assign(self.weight, new_weight)
                                
        sess.run(update_neighbor)
                                
    def train(self, dataset, num_epochs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                                
            for epoch in range(num_epochs):
                print(f'Epoch {epoch}:', end = '')
                for data in dataset:
                    dic = {self.input: data}
                    sess.run(self.update, feed_dict = dic)
                print('Done')
            location = sess.run(self.location)
            weight = sess.run(self.weight)
                                
            for i, loc in enumerate(location):
                self.cluster[int(loc[0])].append(weight[i])


# In[3]:


df = pd.read_csv('clustering.csv')
df


# <h3>Feature Selection</h3>

# In[4]:


data = df[['bruises','odor','stalk-shape','veil-type','spore-print-color']]
data


# In[5]:


data.odor = data.odor.map( {'a': 1, 'l': 2, 'c': 3, 'y': 4, 'f': 5, 'm': 6, 'n': 7, 'p': 8, 's': 9} )
data['stalk-shape'] = data['stalk-shape'].map( {'e': 1, 't': 2} )
data['veil-type'] = data['veil-type'].map( {'p': 1, 'u': 2} )
data


# <h3>Feature Extraction</h3>

# In[6]:


data = data.to_numpy()
data


# In[7]:


data = StandardScaler().fit_transform(data)
data


# In[8]:


principalComponents = PCA(n_components=3).fit_transform(data)
principalComponents


# <h3>Training</h3>

# In[9]:


som = SOM(16, 16, 3, 0.1)
som.train(principalComponents, 2500)


# <h3>Visualization</h3>

# In[10]:


plt.imshow(som.cluster)
plt.show()

