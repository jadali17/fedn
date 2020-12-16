import keras
from keras.layers import LSTM, Dense,Reshape
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import requests
import os


import tempfile


# Create an initial CNN Model
def create_seed_model():
        print(tf.__version__)
        model = Sequential()
        model.add(LSTM(5, input_shape=(5, 4),return_sequences=False, stateful=False))
        model.add(Dense(1))
   
        model.compile(loss = "mae", optimizer = 'adam',metrics=['mae'])
        return model


if __name__ == '__main__':
        print(tf.__version__)
	# Create a seed model and push to Minio
	#model = create_seed_model()
	#runtime = AllianceRuntimeClient()
	#model_id = runtime.set_model(pickle.dumps(model))
	#print("Created seed model with id: {}".format(model_id))
	
        # Create a seed model and push to Minio
        model = create_seed_model()
        outfile_name = "../m2314.h5"
        #fod, outfile_name = tempfile.mkstemp(suffix='.h5')
        model.save(outfile_name, save_format='h5')

        #project = Project()
        #from scaleout.repository.helpers import get_repository

        #repo_config = {'storage_access_key': 'minio',
        #                           'storage_secret_key': 'minio123',
        #                           'storage_bucket': 'models',
        #                           'storage_secure_mode': False,
        #                           'storage_hostname': 'minio',
        #                           'storage_port': 9000}
        #storage = get_repository(repo_config)

        #model_id = storage.set_model(outfile_name,is_file=True)
        #os.unlink(outfile_name)
        #print("Created seed model with id: {}".format(model_id)) 
        

