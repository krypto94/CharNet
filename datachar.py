import numpy as np
from scipy.misc import imread,imresize,imshow
import os

''' this script reads the dataset downloaded pre-processes it and converts it into suitable binary format'''
    		

k=1
j=1

for m in xrange(62):
	if(j<10):
		path = "/home/rishabh/Documents/English/Img/GoodImg/Bmp/Sample00"+str(j)# change the path
		
	elif(j >9):
		path = "/home/rishabh/Documents/English/Img/GoodImg/Bmp/Sample0"+str(j)
		
	F = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
	
	sample001 = np.zeros((F,3072))

	for i in xrange(F):
		if(k < 10 and j < 10):
			img=imread(path+"/img00"+str(j)+"-0000"+str(k)+".png")
		elif(k < 10 and j > 9):
			img=imread(path+"/img0"+str(j)+"-0000"+str(k)+".png")


		elif(k > 9 and k <100 and j < 10):

			img=imread(path+"/img00"+str(j)+"-000"+str(k)+".png")
		elif(k > 9 and k <100 and j > 9):

			img=imread(path+"/img0"+str(j)+"-000"+str(k)+".png")

		elif(k > 99 and j < 10):
			img=imread(path+"/img00"+str(j)+"-00"+str(k)+".png")
		elif(k > 99 and j > 9):
			img=imread(path+"/img0"+str(j)+"-00"+str(k)+".png")	
		
		


		k=k+1
		img_mean = np.mean(img)#Mean center image
		img = img - img_mean
		img /= np.std(img)#Normalize each image
		gray_img = np.zeros(((img.shape[0],img.shape[1],3)))
		if(len(img.shape)  == 2):#take care of gray scale images in the data base
			gray_img[:,:,0] = img
			img = gray_img
				
		sample001[i] = imresize(img,(32,32,3)).reshape(1,3072)
	k=1	
	imshow(sample001[25].reshape(32,32,3))

	f = file("/home/rishabh/Desktop/TensorFlow/Datachar/label"+str(j)+".bin","wb")
	np.save(f,sample001)
	j=j+1
