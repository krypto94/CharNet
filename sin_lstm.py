from keras.models import Sequential  
from keras.layers.core import Dense, Activation ,Dropout 
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization as bn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
	df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/sin.csv")
	def _load_data(data, n_prev = 100):  
		docX = []
		docY = []
    		for i in range(len(data)-n_prev):
        		docX.append(data.iloc[i:i+n_prev].as_matrix())
        		docY.append(data.iloc[i+n_prev].as_matrix())
    		alsX = np.array(docX)
    		alsY = np.array(docY)
    		return alsX,alsY


	x_train,y_train = _load_data(df.iloc[:1000,:],100)
	print x_train.shape


	model = Sequential()
	model.add(LSTM(10,input_shape=(100,1))) 
	model.add(bn()) 
	model.add(Dropout(0.6))  
	model.add(Dense(1))  
	model.add(Activation("tanh"))  
	model.compile(loss="mean_squared_error", optimizer="rmsprop")  

	history = model.fit(x_train, y_train, batch_size=50, nb_epoch=20, validation_split=0.1)  
except KeyboardInterrupt:
	print "Training Stopped.. Please Wait For Loss Plots"
	

model_json = model.to_json()
with open("model_sin.json", "w") as json_file:
    json_file.write(model_json)
	# serialize weights to HDF5
model.save_weights("model_sin.h5")
print("Saved model to disk")
	
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()