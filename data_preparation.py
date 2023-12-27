import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
print(data.head())

## Split data between data and train 
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

dev_set = data[0:1500].T
Y_dev = dev_set[0]
X_dev = dev_set[1:n]
X_dev = X_dev / 255.0

training_set = data[1500:m].T
Y_train = training_set[0]
X_train = training_set[1:n]
X_train = X_train / 255.0
_,m_train = X_train.shape