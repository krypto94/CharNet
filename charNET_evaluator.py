import tensorflow as tf 
import numpy as np 
import cPickle
from scipy.misc import imshow,imresize,imread


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


# f=file("/home/rishabh/Desktop/TensorFlow/Datachar(Pre-Processed)/DATA.bin","rb")
# dataset=np.load(f) 
# f.close()
# d=np.split(dataset,[3072],axis = 1)
# data = d[0]
# labels =d[1]
# train_data = np.split(data,[7000])[0]
# test_data  = np.split(data,[7000])[1]		
# train_labels = np.split(labels,[7000])[0]
# test_labels  = np.split(labels,[7000])[1]


x=tf.placeholder(tf.float32,shape=[None,3072])
# y=tf.placeholder(tf.float32,shape=[None,62])


x_image = tf.reshape(x,[-1,32,32,3])

w_conv1 = weight_variable([3,3,3,20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

w_conv2 = weight_variable([3,3,20,30])
b_conv2 = bias_variable([30])
h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2)+b_conv2) 

pool1 = max_pool(h_conv2)

w_conv3 = weight_variable([3,3,30,40])
b_conv3 = bias_variable([40])
h_conv3 = tf.nn.relu(conv2d(pool1,w_conv3)+b_conv3) 

w_conv4 = weight_variable([3,3,40,50])
b_conv4 = bias_variable([50])
h_conv4 = tf.nn.relu(conv2d(h_conv3,w_conv4)+b_conv4)

pool2 = max_pool(h_conv4)

w_conv5 = weight_variable([3,3,50,60])
b_conv5 = bias_variable([60])
h_conv5 = tf.nn.relu(conv2d(pool2,w_conv5)+b_conv5)


W_fc1 = weight_variable([8*8*60,300 ])#fully connected layer
b_fc1 = bias_variable([300])

h_pool2_flat = tf.reshape(h_conv5, [-1, 8*8*60])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([300, 62])
b_fc2 = bias_variable([62])

scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2	


# learning_rate = tf.placeholder(tf.float32)

probs = tf.nn.softmax(scores)
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, y))
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(scores,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
sess=tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
saver.restore(sess, "/home/rishabh/Desktop/TensorFlow/CharNET MODEL/model.ckpt")
print("Model restored.")



# k=0
# batch_size = 25
# m=1
# L=1e-3
# for i in xrange(70000):
# 	sess.run(train_step,feed_dict={x:data[k:k+batch_size],y:labels[k:k+batch_size],keep_prob : 0.5,learning_rate:L})
# 	if(i%100 == 0):
# 		train_accuracy = accuracy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L})
# 		print("step %d, training accuracy%g\n"%(i, train_accuracy))
# 		loss = cross_entropy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L})
# 		print "loss is\n",loss
# 	k=k+batch_size
# 	if(k>=7000):
# 		test_accuracy = accuracy.eval(feed_dict={x:test_data, y:test_labels,keep_prob:1.0,learning_rate:L})
# 		print "epoch Finished, Test set accuracy is",float(test_accuracy),m
# 		k=0
# 		m=m+1
# 		save_path = saver.save(sess, "/home/rishabh/Desktop/TensorFlow/CharNET MODEL/model.ckpt")
#   		print("Model saved in file: %s" % save_path)
# 	if(m%100 == 0):
# 		print "Decaying learning rate 10 FOLDS"
# 		l=L/10


a={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',
	23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',
	42:'g',43:'h',44:'i',45:'j',46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',61:'z'}
	
img=imread('/home/rishabh/Documents/English/Img/GoodImg/Bmp/Sample003/img003-00013.png')
test_img=imresize(img,(32,32,3))
test_img=test_img.reshape(1,3072)
probability = probs.eval(feed_dict={x:test_img,keep_prob:1.0})
# print probability

print ([a[i] for i in np.argsort(probability)[0,-10:]][::-1])


def getActivation(layer,stimuli,N):
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