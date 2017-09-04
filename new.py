from keras.models import Sequential  
from keras.layers.core import Dense, Activation ,Dropout 
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization as bn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

try:
	df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv")

	temp = df.as_matrix()
	temp =temp.astype(float)
	scaler = MinMaxScaler(feature_range=(0, 1))
	temp = scaler.fit_transform(temp)
	df=pd.DataFrame(temp)


	def _load_data(data, n_prev = 100):  
		docX = []
		docY = []
    		for i in range(len(data)-n_prev):
        		docX.append(data.iloc[i:i+n_prev].as_matrix())
        		docY.append(data.iloc[i+n_prev].as_matrix())
    		alsX = np.array(docX)
    		alsY = np.array(docY)
    		return alsX,alsY

        n = 10		
	x_train,y_train = _load_data(df.iloc[:4000,:],n)



	model = Sequential()
	model.add(LSTM(100,input_shape=(n,15),return_sequences=True)) 
	model.add(bn())  
	model.add(LSTM(50, return_sequences=False)) 
	model.add(bn())   
	model.add(Dense(100)) 
	model.add(Dropout(0.0))
	model.add(Dense(15))  
	model.add(Activation("linear"))  
	model.compile(loss="mean_squared_error", optimizer="adam")  

	history = model.fit(x_train, y_train, batch_size=25, nb_epoch=20, validation_split=0.2)  
except KeyboardInterrupt:
	print "Training Stopped.. Please Wait For Loss Plots"
tempp = np.zeros(((1,n,15)))
tempp[0,:,:] =  temp[490:500,:]

for i in range(50):
	tempp[0,:,:] =  temp[490+i:500+i,:]
	trainPredict = model.predict(tempp)
	trainPredict = scaler.inverse_transform(trainPredict)
	print np.round(trainPredict)
	temppp = np.zeros((1,15))
	temppp = temp[500+i,:]
	print scaler.inverse_transform(temppp.reshape(1,-1))	
	print "==============================================="
print "INTIATING AUTO_TUNE PROTOCOL"
print "==================================="
x_train,y_train = _load_data(df.iloc[10000:12000,:],n)

history = model.fit(x_train, y_train, batch_size=25, nb_epoch=100, validation_split=0.2)  

tempp = np.zeros(((1,n,15)))
tempp[0,:,:] =  temp[15000:15010,:]

for i in range(50):
	tempp[0,:,:] =  temp[15000+i:15010+i,:]
	trainPredict = model.predict(tempp)
	trainPredict = scaler.inverse_transform(trainPredict)
	print np.round(trainPredict)
	temppp = np.zeros((1,15))
	temppp = temp[15010+i,:]
	print scaler.inverse_transform(temppp.reshape(1,-1))	
	print "==============================================="




model_json = model.to_json()
with open("model_norm_mean.json", "w") as json_file:
    json_file.write(model_json)
	# serialize weights to HDF5
model.save_weights("model_norm_mean.h5")
print("Saved model to disk")
	
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
