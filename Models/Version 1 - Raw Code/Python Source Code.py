import numpy as np
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

df.drop(["RowNumber" , "CustomerId" , "Surname"] , axis = 1 , inplace = True)

data =pd.get_dummies(df , columns = ["Geography" , "Gender"] , drop_first = True)

train , test = np.split(data.sample(frac = 1) , [int(0.8*len(df))])

from sklearn.preprocessing import StandardScaler

def pre(dataframe):
    
    x = dataframe.drop("Exited" , axis = 1)
    y = dataframe["Exited"]
    
    sc = StandardScaler()
    
    sc.fit_transform(x , y)
    
    return x , y
    
X_train , Y_train = pre(train)
X_test , Y_test = pre(test)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(11 , activation = "relu" , input_dim = 11))
model.add(Dense(11 , activation = "relu" , input_dim = 11))
model.add(Dense(1 , activation = "sigmoid"))

model.compile(loss = "binary_crossentropy" , optimizer = "Adam" , metrics = ["accuracy"])

hitstory = model.fit(X_train , Y_train , epochs = 100 , validation_split = 0.2)

y_log = model.predict(X_test)
y_pred = np.where(y_log > 0.5 , 1 , 0)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test , y_pred))
