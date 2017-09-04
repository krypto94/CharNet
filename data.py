import pandas as pd 
import numpy as np 
'''
NOTE : UNCOMMENT ONLY ONE BLOCK AT A TIME
DATE is generated randomly, but data is placed in the matix sequentially
The Customer is assumed to make a tranaction between 21 - 29 of every month,
The green range of tranaction is $200 - $300,For model simplicity each month the 
trancation is made to beneficiary 54 via bank 4,the rand amount and job and status code
are completely random. numpy.random.randint draws value from a gaussian distribution each
time its called.For adding noise to the system, other dates are generated with are either close
or in some cases overlap with the target tranaction dates, this is done to keep this custom
dataset close to the real world scenario and also this will insure that the model will NEVER have
100% precision recall.
This dataset has 20000 rows with the following 16 Fields :
DATE , JOB1, JOB2, JOB3, JOB4, NEFT, RTGS, IFT, RTGUC, AMOUNT, BENEFICIARY NAME, CITY NAME
ACCOUNT NUMBER, BANK, STAGE CODE1, STAGE CODE2.
For training an LSTM based prediction model, this a small dataset and should be used for demonstration
purposes only. Through this dataset, I will try to prove that LSTM can extract regular patterns from
a Periodic transactions, and thus they can be used for catching anomalies such as fraud, malfunctions etc.
in a given tranction.
'''

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",header = None)
temp = df.as_matrix()
temp =temp.astype(int)
dataset = np.empty((temp.shape[0],temp.shape[1]+1+1))
############################################################################################
# date = np.empty((20000,1))
# flg = 0
# for i in range(5000):
	
# 	date1 = 20 + np.random.randint(8)
# 	date2 = np.random.randint(1,5)
# 	date3 = np.random.randint(5,10)
# 	date4 = np.random.randint(10,18)
# 	date[flg+0,0] = date2
# 	date[flg+1,0] = date3
# 	date[flg+2,0] = date4
# 	date[flg+3,0] = date1
# 	flg = flg + 4
# 	if(flg == 19998):
# 		break	 
# # date[19996,0] = 6
# # date[19997,0] = 15
# # date[19998,0] = 24
# # date[19999,0] = 3
# df1 = pd.DataFrame(date)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header = False)
# ##################### date generator ######################################################

###########################################################################################
# job = np.zeros((20000,4))
# flg=0
# for i in range(20000):
# 	job[i,flg ] = 1
# 	flg = flg +1
# 	if(flg == 4):
# 		flg = 0
# dataset[:,0] = temp.T
# dataset[:,1:5] = job
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header=False)
##################### JOB GENERATOR #######################################################

###########################################################################################
# MODE = np.zeros((20000,4))
# flg=0
# for i in range(20000):
# 	MODE[i,flg ] = 1
# 	flg = flg +1
# 	if(flg == 4):
# 		flg = 0
	
# dataset[:,:5] = temp
# dataset[:,5:9] = MODE
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False)
##################### MODE GENERATOR ######################################################

###########################################################################################
# amount = np.zeros((20000,1))
# flg = 0
# for i in range(5000):
# 	a=np.random.rand(1)
# 	amount[flg]   = np.round((np.sin(np.pi/8)+a)*100)
# 	amount[flg+1] = np.round((np.sin(np.pi/6)+a)*150)
# 	amount[flg+2] = np.round((np.sin(np.pi/4)+a)*200)
# 	amount[flg+3] =np.round((np.sin(np.pi/2.1)+a)*500)
# 	flg=flg+4	
# dataset[:,:9] = temp
# dataset[:,9] = amount.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header=False)
##################### AMOUNT GENERATOR ######################################################

############################################################################################
# NAME = np.zeros((20000,1))
# flg=0
# for i in range(20000):
# 	NAME[i] = flg
# 	flg=flg+1
# 	if(flg == 10):
# 		flg=0
	
# dataset[:,:10] = temp
# dataset[:,10] = NAME.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header =  False)
##################### BENEFICIARY NAME GENERATOR ############################################

#############################################################################################
# NAME = np.zeros((20000,1))
# flg=0
# for i in range(20000):
# 	NAME[i] = flg
# 	flg=flg+1
# 	if(flg == 20):
# 		flg=0

# dataset[:,:11] = temp
# dataset[:,11] = NAME.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header=False)
##################### CITY GENERATOR ########################################################

#############################################################################################
# NUMBER = np.zeros((20000,1))
# for i in range(20000):
# 	rand = np.random.randint(500)
# 	if(i%3 == 0):
# 		NUMBER[i,0] = 237
# 	else:
# 		NUMBER[i,0] = rand
# dataset[:,:14] = temp
# dataset[:,12] = NUMBER.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### ACCOUNT NUMBER GENERATOR ##############################################

#############################################################################################
# BANK = np.zeros((20000,1))
# flg=0
# for i in range(20000):
# 	BANK[i] = flg
# 	flg=flg+1
# 	if(flg == 5):
# 		flg=0
# dataset[:,:12] = temp
# dataset[:,12] = BANK.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header=False)
##################### BANK GENERATOR #######################################################

###########################################################################################
STAGE = np.zeros((20000,2))
for i in range(20000):
	rand = np.random.randint(2)
	if(i%3 == 0):
		STAGE[i,1] = 1
	else:
		STAGE[i,rand] = 1
dataset[:,:13] = temp
dataset[:,13:15] = STAGE
df1 = pd.DataFrame(dataset)
df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset_sin.csv",index = False,header=False)
##################### STAGE GENERATOR #######################################################