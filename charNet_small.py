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


f=file("/home/rishabh/Desktop/TensorFlow/Datachar(Pre-Processed)/DATA.bin","rb")
dataset=np.load(f) 
f.close()
d=np.split(dataset,[3072],axis = 1)
data = d[0]
labels =d[1]
train_data = np.split(data,[7000])[0]
test_data  = np.split(data,[7000])[1]		
train_labels = np.split(labels,[7000])[0]
test_labels  = np.split(labels,[7000])[1]


x=tf.placeholder(tf.float32,shape=[None,3072])
y=tf.placeholder(tf.float32,shape=[None,62])
x_image = tf.reshape(x,[-1,32,32,3])

w_conv1 = weight_variable([3,3,3,30])
b_conv1 = bias_variable([30])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

#NORM
norm1 = tf.nn.lrn(h_conv1,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')


w_conv2 = weight_variable([3,3,30,40])
b_conv2 = bias_variable([40])
h_conv2 = tf.nn.relu(conv2d(norm1,w_conv2)+b_conv2) 

pool1 = max_pool(h_conv2)
keep_prob_pool1 = tf.placeholder(tf.float32)
pool1drop = tf.nn.dropout(pool1, keep_prob_pool1)

#norm
norm2 = tf.nn.lrn(pool1drop,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

w_conv3 = weight_variable([3,3,40,60])
b_conv3 = bias_variable([60])
h_conv3 = tf.nn.relu(conv2d(norm2,w_conv3)+b_conv3) 

norm3 = tf.nn.lrn(h_conv3,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')


pool2 = max_pool(norm3)


W_fc1 = weight_variable([8*8*60,1000 ])#fully connected layer
b_fc1 = bias_variable([1000])

h_pool2_flat = tf.reshape(pool2, [-1, 8*8*60])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1000, 62])
b_fc2 = bias_variable([62])

scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2	


learning_rate = tf.placeholder(tf.float32)

probs = tf.nn.softmax(scores)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(scores,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
sess=tf.InteractiveSession()
sess.run(tf.initialize_all_variables())




k=0
batch_size = 20
m=1
L=1e-3
flag = 0
for i in xrange(70000):
	sess.run(train_step,feed_dict={x:data[k:k+batch_size],y:labels[k:k+batch_size],keep_prob :0.9,learning_rate:L,keep_prob_pool1:1.0})
	if(i%100 == 0):
		train_accuracy = accuracy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0})
		print("step %d, training accuracy%g\n"%(i, train_accuracy))
		loss = cross_entropy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0})
		print "loss is\n",loss
	k=k+batch_size
	if(k>=7000):
		test_accuracy = accuracy.eval(feed_dict={x:test_data, y:test_labels,keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0})
		print "epoch Finished, Test set accuracy is",float(test_accuracy),m
		k=0
		m=m+1
		flag = flag +1
		save_path = saver.save(sess, "/home/rishabh/Desktop/TensorFlow/CharNET MODEL/model.ckpt")
  		print("Model saved in file: %s" % save_path)
	if(m%20 == 0 and flag > 0):
		print "Decaying learning rate 10 FOLDS"
		L=L/10
		print L
		flag = 0




# img=imread('/home/rishabh/Documents/8e427328807052.56053ef96e121.jpg')
# test_img=imresize(img,(32,32,3))
# def getActivation(layer,stimuli,N):
# 	units=layer.eval(session=sess,feed_dict={x:np.reshape(stimuli,[1,3072],order="F"),keep_prob:1.0})
# 	print units.shape
# 	plotNNfilter(units,N)
# def plotNNfilter(units,N):
# 	filters = units.shape[3]
# 	for i in xrange(0,N):
# 		img=imresize((units[0,:,:,i]),(200,200))
# 		imshow(img)



# getActivation(h_conv1,test_img,6)
# getActivation(h_conv3,test_img,6)
# getActivation(h_conv5,test_img,6)



sess.close()