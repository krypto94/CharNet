import pandas as pd 
import numpy as np 


df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_string.csv",header = None)
temp = df.as_matrix()
# temp =temp.astype(int)
dataset = np.zeros((20000,15),dtype = int)
dataset[:,0] = temp[:,0]



cust ={}
city={}
bank={}

for i in range(20000) :
	if temp[i,1] == 'JOB1':
		dataset[i,1] = 1
	if temp[i,1] == 'JOB2':
		dataset[i,2] = 1
	if temp[i,1] == 'JOB3':
		dataset[i,3] = 1
	if temp[i,1] == 'JOB4':
		dataset[i,4] = 1

for i in range(20000) :
	if temp[i,2] == 'NEFT':
		dataset[i,5] = 1
	if temp[i,2] == 'RTGS':
		dataset[i,6] = 1
	if temp[i,2] == 'IFT':
		dataset[i,7] = 1
	if temp[i,2] == 'CASH':
		dataset[i,8] = 1					
dataset[:,9] = temp[:,3]
cflg = 0
ciflg = 0
bflg = 0

for i in range(20000):
	if(not (temp[i,4] in cust.keys())):
		cust[temp[i,4]] = cflg
		cflg = cflg + 1
	if(not (temp[i,5] in city.keys())):
		city[temp[i,5]] = ciflg
		ciflg = ciflg + 1
	if(not (temp[i,6] in bank.keys())):
		bank[temp[i,6]] = bflg
		bflg = bflg + 1	
for i in range(20000):
	dataset[i,10] = cust[temp[i,4]]
	dataset[i,11] = city[temp[i,5]]
	dataset[i,12] = bank[temp[i,6]]

for i in range(20000):
	if temp[i,7] == 'DEBIT':
		dataset[i,14] = 1
	if temp[i,7] == 'CREDIT':
		dataset[i,13] = 1
print ciflg
print cflg
print bflg
df=pd.DataFrame(dataset)
# df.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_regen.csv",index = False,header=False)
print "INITIATING COMPARE ROUTINE"

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",header = None)
temp = df.as_matrix()
temp =temp.astype(int)
print np.array_equal(dataset,temp)

flg = 0
for i in range(20000):
	for j in range(15):
		if(not(dataset[i,j] == temp[i,j])):
			print temp[i,5:9]
			print dataset[i,5:9]
			flg = flg +1
print flg			