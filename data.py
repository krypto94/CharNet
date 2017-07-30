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

df=pd.read_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv")
temp = df.as_matrix()
dataset = np.empty((temp.shape[0],temp.shape[1]))
############################################################################################
# date = np.empty((20000,1))
# flg = 0
# date[0,0] = 1
# for i in range(7000):
	
# 	date1 = 21 + np.random.randint(8)
# 	date2 = np.random.randint(1,14)
# 	# date3 = np.random.randint(24,31)
# 	date4 = np.random.randint(14,17)
# 	date[flg+1,0] = date2
# 	date[flg+2,0] = date4
# 	date[flg+3,0] = date1
# 	date[flg+4,0] = date2
# 	flg = flg + 3
# 	if(flg == 19998):
# 		break	 
# date[19996,0] = 6
# date[19997,0] = 15
# date[19998,0] = 24
# date[19999,0] = 3

# dataset[:,:10] = temp
# dataset[:,0] = date.T

# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### DATE GENERATOR ######################################################

###########################################################################################
# job = np.zeros((20000,4))
# for i in range(20000):
# 	rand = np.random.randint(3)
# 	if(i%3 == 0):
# 		job[i,3] = 1
# 	else:
# 		job[i,rand] = 1
# dataset[:,0] = temp.T
# dataset[:,1:5] = job
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### JOB GENERATOR #######################################################

###########################################################################################
# MODE = np.zeros((20000,4))
# for i in range(20000):
# 	rand = np.random.randint(3)
# 	if(i%3 == 0):
# 		MODE[i,3] = 1
# 	else:
# 		MODE[i,rand] = 1
# dataset[:,:5] = temp
# dataset[:,5:9] = MODE
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### MODE GENERATOR ######################################################

###########################################################################################
# AMOUNT = np.zeros((20000,1))
# for i in range(20000):
# 	rand = np.random.randint(200,300)
# 	rand2 = np.random.randint(10,150)
# 	rand3 = np.random.randint(350,450)
# 	if(i%3 == 0):
# 		AMOUNT[i,0] = rand
# 	elif(i%7 == 0):
# 		AMOUNT[i,0] = rand2
# 	else:
# 		AMOUNT[i,0] = rand3	
# dataset[:,:9] = temp
# dataset[:,9] = AMOUNT.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### AMOUNT GENERATOR ######################################################

#############################################################################################
# NAME = np.zeros((19999,1))
# for i in range(19999):
# 	rand = np.random.randint(100)
# 	if(i%3 == 0):
# 		NAME[i,0] = 54
# 	else:
# 		NAME[i,0] = rand
# dataset[:,:16] = temp
# dataset[:,10] = NAME.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset1.csv",index = False)
##################### BENEFICIARY NAME GENERATOR ############################################

#############################################################################################
# NAME = np.zeros((20000,1))
# for i in range(20000):
# 	rand = np.random.randint(10)
# 	if(i%3 == 0):
# 		NAME[i,0] = 7
# 	else:
# 		NAME[i,0] = rand
# dataset[:,:11] = temp
# dataset[:,11] = NAME.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
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
# for i in range(20000):
# 	rand = np.random.randint(5)
# 	if(i%3 == 0):
# 		BANK[i,0] = 4
# 	else:
# 		BANK[i,0] = rand
# dataset[:,:15] = temp
# dataset[:,13] = BANK.T
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### BANK GENERATOR #######################################################

###########################################################################################
# STAGE = np.zeros((20000,2))
# for i in range(20000):
# 	rand = np.random.randint(2)
# 	if(i%3 == 0):
# 		STAGE[i,1] = 1
# 	else:
# 		STAGE[i,rand] = 1
# dataset[:,:16] = temp
# dataset[:,14:16] = STAGE
# df1 = pd.DataFrame(dataset)
# df1.to_csv("/home/rishabh/Desktop/DeepLearning/keras/Nucleus/dataset.csv",index = False)
##################### STAGE GENERATOR #######################################################