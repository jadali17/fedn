import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


import keras
import numpy
def read_data(filename):

    """ Helper function to read and preprocess data for training with Keras. """
    data = pd.read_csv(filename, sep = ',',index_col=[0])

    data_set=data[['PM10','NO2','PM2_5','NOX']]
    X = []
    y = []
    for i in range(5, len(data_set)):
        X.append(data_set.iloc[i-5:i].values)
        y.append(data_set['PM10'].iloc[i])

    X, y = np.array(X), np.array(y)
    

    return  (X, y)

