import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt



a = range(1000)
b = np.sin(a)
data = np.empty((1000,2))
data[:,0] = a
data[:,1] = b
df = pd.DataFrame(data)
df.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/sin.csv",header = False,index=False)



plt.plot(a,b)
plt.show()

