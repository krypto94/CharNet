from keras.models import Sequential  
from keras.layers.core import Dense, Activation ,Dropout 
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization as bn
import numpy as np
import pandas as pd


df=pd.read_csv("/home/rishabh/Desktop/keras/Nucleus/creditcard_1.csv", header = None)

def _load_data(data, n_prev = 100):  
 
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


x_train,y_train = _load_data(df.iloc[:50000,:],50)



model = Sequential()  
model.add(LSTM(29, input_shape=(50,29),return_sequences=True)) 
model.add(bn())  
model.add(LSTM(70, return_sequences=False)) 
model.add(bn()) 
model.add(Dropout(0.25))  
model.add(Dense(29))  
model.add(Activation("sigmoid"))  
model.compile(loss="mean_squared_error", optimizer="adam")  
model.fit(x_train, y_train, batch_size=15, nb_epoch=10, validation_split=0.05)  


---------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

df = pd.read_csv("/home/rishabh/Desktop/keras/Nucleus/creditcard.csv")
temp = df.as_matrix()

fraud = np.zeros((492,31))
list = []
index = 0
temp[: , 0:30] /= np.std( temp[:,0:30] ,axis = 0)

for i in range(temp.shape[0]):
	if (temp[i,30] == 1.0):
		fraud[index,:] = temp[i,:]
		list.append(i)
		print index
		index = index + 1 
temp = np.delete(temp,list,axis = 0)
df= pd.DataFrame(temp[:,1:30])
df.to_csv("/home/rishabh/Desktop/keras/Nucleus/creditcard_1.csv",header = False,index = False)
df= pd.DataFrame(fraud[:,1:30])	
df.to_csv("/home/rishabh/Desktop/keras/Nucleus/creditcard_fraud.csv",header = False,index = False)
