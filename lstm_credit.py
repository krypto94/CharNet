from keras.models import Sequential  
from keras.layers.core import Dense, Activation ,Dropout 
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization as bn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
	df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_norm_mean.csv")
	def _load_data(data, n_prev = 100):  
		docX = []
		docY = []
    		for i in range(len(data)-n_prev):
        		docX.append(data.iloc[i:i+n_prev].as_matrix())
        		docY.append(data.iloc[i+n_prev].as_matrix())
    		alsX = np.array(docX)
    		alsY = np.array(docY)
    		return alsX,alsY


	x_train,y_train = _load_data(df.iloc[:200,:],100)



	model = Sequential()
	model.add(LSTM(30,input_shape=(100,16),return_sequences=True)) 
	model.add(bn())
	model.add(Dropout(0.25))  
	model.add(LSTM(20, return_sequences=False)) 
	model.add(bn()) 
	model.add(Dropout(0.5))  
	model.add(Dense(16))  
	model.add(Activation("linear"))  
	model.compile(loss="mean_squared_error", optimizer="rmsprop")  

	history = model.fit(x_train, y_train, batch_size=25, nb_epoch=100, validation_split=0.1)  
except KeyboardInterrupt:
	print "Training Stopped.. Please Wait For Loss Plots"
	

model_json = model.to_json()
with open("model_norm_mean.json", "w") as json_file:
    json_file.write(model_json)
	# serialize weights to HDF5
model.save_weights("model_norm_mean.h5")
print("Saved model to disk")
	
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()