import pandas as pd 
import numpy as np 

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",header = None)
temp = df.as_matrix()
temp =temp.astype(int)

dataset =  np.zeros((20000,8),dtype = object)
dataset[:,0] = temp[:,0]
for i in range(20000):
	if(temp[i,1] == 1):
		dataset[i,1] = 'JOB1'
	if(temp[i,2] == 1):
		dataset[i,1] = 'JOB2'
	if(temp[i,3] == 1):
		dataset[i,1] = 'JOB3'
	if(temp[i,4] == 1):
		dataset[i,1] = 'JOB4'			 

for i in range(20000):
	if(temp[i,5] == 1):
		dataset[i,2] = 'NEFT'
	if(temp[i,6] == 1):
		dataset[i,2] = 'RTGS'
	if(temp[i,7] == 1):
		dataset[i,2] = 'IFT'
	if(temp[i,8] == 1):
		dataset[i,2] = 'CASH'

dataset[:,3]= temp[:,9]
customer=['Nucleus','Google','Apple','Tesla','Qualcomm','Intel','SpaceX','Reliance','Xiaomi','Airtel']
for i in  range(20000):
	dataset[i,4]=customer[temp[i,10]]
city = ['Noida','Delhi','Mumbai','Chennai','Lucknow','Kolkata','Bangalore','Boston','NY','Singapore','London','Mexico','Moscow','Taiwan',
'Tokyo','Miami','Nepal','Assam','Goa','Hyderabad']
for i in  range(20000):
	dataset[i,5]=city[temp[i,11]]
bank = ['ICICI','SBI','HDFC','CITI','RBI']

for i in  range(20000):
	dataset[i,6]=bank[temp[i,12]]
for i in range(20000):
	if(temp[i,13] == 1):
		dataset[i,7] = 'CREDIT'
	if(temp[i,14] == 1):
		dataset[i,7] = 'DEBIT'	 	

df=pd.DataFrame(dataset)
df.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_string.csv",index = False,header=False)