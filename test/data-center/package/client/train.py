from __future__ import print_function
import sys
import keras
import tensorflow as tf 
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow.keras.models as krm

from read_data import read_data


def train(model,data):
    print("-- RUNNING TRAINING --")



    # The data, split between train and test sets
    (X_train, y_train) = read_data(data)
    

    history=model.fit(X_train, y_train, epochs=30, batch_size=72, verbose=0)
    print(history.history['loss'])
    print("-- TRAINING COMPLETED --")
    return model

if __name__ == '__main__':
    # Read the model
    model = krm.load_model(sys.argv[1])
    model = train(model,'../data/train.csv')
    model.save(sys.argv[2])



