import numpy as np
import pandas as pd 
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt


df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/sin.csv")
print df







t=range(899)

# temp = df.as_matrix()
# X_test = np.zer((1,50,16)))
# X_test[0,:,:] = temp[1:51,:]
# y_test = temp[52,:]


json_file = open('model_sin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_sin.h5")
print("Loaded model from disk")

tempo = df.as_matrix()

temp = np.zeros(((1,999,1)))
temp[0,:,0] = tempo.T
x_test = np.zeros(((1,100,1)))
predicted = np.zeros(899)
for i in range(899):
	x_test[0,:,0] = temp[0,i:i+100,0]
	# print i
	# print x_test.shape
	predicted[i] = loaded_model.predict(x_test)



# print t.shape
print predicted.shape
# rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
# plt.plot(t,df.as_matrix())
print predicted[:50]
plt.plot(t[:100],predicted[:100])
plt.show()
print "::::::::::::::::::::::::::::::::::::::::::::::::::::::::"

