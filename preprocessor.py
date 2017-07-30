import numpy as np 
import pandas as pd 

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv")


temp=df.as_matrix()
temp = temp.astype(float)
temp /= np.std(temp, axis = 0 )
df=pd.DataFrame(temp)
df.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_normalised.csv",index = False)
temp -= np.mean(temp,axis = 0)
df=pd.DataFrame(temp)
df.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_norm_mean.csv",index = False)