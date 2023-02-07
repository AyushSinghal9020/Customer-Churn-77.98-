# **DATA PROCESSING**

import numpy as np # Array Processing
import pandas as pd # Data Processing

# **DATA ANALYSIS**

import matplotlib.pyplot as plt # Plots

# **PRE PROCESSING**

from sklearn.preprocessing import StandardScaler # Scaling of Data
from imblearn.over_sampling import RandomOverSampler # Sampling of Data

# **NEURAL NETWORKS**

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# **METRICS**
from sklearn.metrics import accuracy_score

df = pd.read_csv("Churn_Modelling.csv")

df.drop(["RowNumber" , "CustomerId" , "Surname"] , axis = 1 , inplace = True)

data =pd.get_dummies(df , columns = ["Geography" , "Gender"] , drop_first = True)

train , test = np.split(data.sample(frac = 1) , [int(0.8*len(df))])

def pre(dataframe , oversampling = True):
    
    x = dataframe.drop("Exited" , axis = 1)
    y = dataframe["Exited"]
    
    sc = StandardScaler()
    ros = RandomOverSampler()
    
    sc.fit_transform(x , y)
    
    if oversampling:
        ros.fit_resample(x ,y)
    
X_train , Y_train = pre(train , oversampling = True)
X_test , Y_test = pre(test)

model = Sequential()
model.add(Dense(11 , activation = "relu" , input_dim = 11))
model.add(Dense(11 , activation = "relu" , input_dim = 11))
model.add(Dense(1 , activation = "sigmoid"))

model.compile(loss = "binary_crossentropy" , optimizer = "Adam" , metrics = ["accuracy"])
hitstory = model.fit(X_train , Y_train , epochs = 100 , validation_split = 0.2)

y_log = model.predict(X_test)
y_pred = np.where(y_log > 0.5 , 1 , 0)
print(accuracy_score(Y_test , y_pred))
