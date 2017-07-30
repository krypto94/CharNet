import numpy as np
import pandas as pd 
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_norm_mean.csv")
# def _load_data(data, n_prev = 100):  
# 	docX = []
# 	docY = []
#     for i in range(len(data)-n_prev):
#         docX.append(data.iloc[i:i+n_prev].as_matrix())
#         docY.append(data.iloc[i+n_prev].as_matrix())
#     	alsX = np.array(docX)
#     	alsY = np.array(docY)
#     	return alsX,alsY
# x_train,y_train = _load_data(df.iloc[:20000,:],50)
temp = df.as_matrix()
X_test = np.zeros(((1,100,16)))
X_test[0,:,:] = temp[10:110,:]
y_test = temp[110,:]
# print temp[6]

real = [29,	1,	0	,0,	0,	0	,0	,0,	1	,400,	46,	7,	237,	4	,0	,1]
real /= np.std(real)
real -= np.mean(real) 


json_file = open('model_norm_mean.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_norm_mean.h5")
print("Loaded model from disk")




predicted = loaded_model.predict(X_test) 
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print rmse[10]
rmse = np.sqrt(((predicted - real) ** 2).mean(axis=0))
print rmse[10]

sum = 0
for i in range(16):
	sum += (predicted[0,i] - real[i])**2

print (np.sqrt(sum))
sum = 0
for i in range(16):
	sum += (predicted[0,i] - y_test[i])**2
print (np.sqrt(sum))


# print predicted
# print "::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
# print rmse