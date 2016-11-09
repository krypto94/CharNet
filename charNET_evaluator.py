import tensorflow as tf 
import numpy as np 
from scipy.misc import imshow,imresize,imread
'''Evaluate the learned model on images'''

def weight_variable(shape):
	initial  = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x,ksize=[1,2,2,1]):
	return	tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')
x=tf.placeholder(tf.float32,shape=[None,3072])
x_image = tf.reshape(x,[-1,32,32,3])

w_conv1 = weight_variable([3,3,3,20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

norm1 = tf.nn.lrn(h_conv1,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

w_conv2 = weight_variable([3,3,20,30])
b_conv2 = bias_variable([30])
h_conv2 = tf.nn.relu(conv2d(norm1,w_conv2)+b_conv2) 

pool1 = max_pool(h_conv2)
keep_prob_pool1 = tf.placeholder(tf.float32)
pool1drop = tf.nn.dropout(pool1, keep_prob_pool1)# Pooling DropOut



#Normalization Layer 2
norm2 = tf.nn.lrn(pool1drop,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')


w_conv3 = weight_variable([3,3,30,40])
b_conv3 = bias_variable([40])
h_conv3 = tf.nn.relu(conv2d(norm2,w_conv3)+b_conv3) 

#Normalization layer 3
norm3 = tf.nn.lrn(h_conv3,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')

w_conv4 = weight_variable([3,3,40,50])
b_conv4 = bias_variable([50])
h_conv4 = tf.nn.relu(conv2d(norm3,w_conv4)+b_conv4)

pool2 = max_pool(h_conv4)
keep_prob_pool2 = tf.placeholder(tf.float32)
pool2drop = tf.nn.dropout(pool2, keep_prob_pool2)



#Normalization layer
norm4 = tf.nn.lrn(pool2drop,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm4')



w_conv5 = weight_variable([3,3,50,60])
b_conv5 = bias_variable([60])
h_conv5 = tf.nn.relu(conv2d(pool2,w_conv5)+b_conv5)

#Normalization layer
norm5 = tf.nn.lrn(h_conv5,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm5')



W_fc1 = weight_variable([8*8*60,300 ])#fully connected layer
b_fc1 = bias_variable([300])

h_pool2_flat = tf.reshape(norm5, [-1, 8*8*60])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([300, 62])
b_fc2 = bias_variable([62])

scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2	
probs = tf.nn.softmax(scores)
saver = tf.train.Saver()
sess=tf.InteractiveSession()
saver.restore(sess, "path/to/model/model.ckpt")#path to the stored ckpt file for restoring model
print("Model restored.")
a={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',
	23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',
	42:'g',43:'h',44:'i',45:'j',46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',61:'z'}
#Dictionary to convert predicted clas to character predicted	
img=imread('path/to/test/image/img.png')#path to test image
test_img=imresize(img,(32,32,3))
test_img=test_img.reshape(1,3072)
probability = probs.eval(feed_dict={x:test_img,keep_prob:1.0})
# print probability //uncomment this to print the test probability

print ([a[i] for i in np.argsort(probability)[0,-10:]][::-1])


def getActivation(layer,stimuli,N):# View the layer activation
	units=layer.eval(session=sess,feed_dict={x:stimuli,keep_prob:1.0})
	print units.shape
	plotNNfilter(units,N)
def plotNNfilter(units,N):
	filters = units.shape[3]
	for i in xrange(0,N):
		img=imresize((units[0,:,:,i]),(200,200))
		imshow(img)



getActivation(h_conv1,test_img,6)
getActivation(h_conv3,test_img,6)
getActivation(h_conv5,test_img,6)



sess.close()
