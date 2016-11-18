import tensorflow as tf 
import numpy as np 

def weight_variable(shape):
	initial  = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')# Stride = 1
def max_pool(x,ksize=[1,2,2,1]):
	return	tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')


f=file("path/to/dataset/DATA.bin","rb") #Change path here to load dataset,Description of the dataset can be found in readme
dataset=np.load(f) 
f.close()
d=np.split(dataset,[3072],axis = 1)
data = d[0]
labels =d[1]
train_data = np.split(data,[7000])[0]
test_data  = np.split(data,[7000])[1]		
train_labels = np.split(labels,[7000])[0]
test_labels  = np.split(labels,[7000])[1]#Datset loaded and split into train an test datasets


x=tf.placeholder(tf.float32,shape=[None,3072])
y=tf.placeholder(tf.float32,shape=[None,62])
x_image = tf.reshape(x,[-1,32,32,3])

w_conv1 = weight_variable([3,3,3,20])#20 Filters in layer 1
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

#Normalization layer
norm1 = tf.nn.lrn(h_conv1,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')


w_conv2 = weight_variable([3,3,20,30])# 30 Filters in layer 2
b_conv2 = bias_variable([30])
h_conv2 = tf.nn.relu(conv2d(norm1,w_conv2)+b_conv2) 

pool1 = max_pool(h_conv2)# 2X2 max pooling
keep_prob_pool1 = tf.placeholder(tf.float32)
pool1drop = tf.nn.dropout(pool1, keep_prob_pool1)# Pooling DropOut



#Normalization Layer 2
norm2 = tf.nn.lrn(pool1drop,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')


w_conv3 = weight_variable([3,3,30,40])# 40 Filters
b_conv3 = bias_variable([40])
h_conv3 = tf.nn.relu(conv2d(norm2,w_conv3)+b_conv3) 


#Normalization layer 3
norm3 = tf.nn.lrn(h_conv3,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')



w_conv4 = weight_variable([3,3,40,50])# 50 filters
b_conv4 = bias_variable([50])
h_conv4 = tf.nn.relu(conv2d(norm3,w_conv4)+b_conv4)

pool2 = max_pool(h_conv4)
keep_prob_pool2 = tf.placeholder(tf.float32)
pool2drop = tf.nn.dropout(pool2, keep_prob_pool2)



#Normalization layer
norm4 = tf.nn.lrn(pool2drop,4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm4')




w_conv5 = weight_variable([3,3,50,60])# 60 Filters
b_conv5 = bias_variable([60])
h_conv5 = tf.nn.relu(conv2d(norm4,w_conv5)+b_conv5)


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


learning_rate = tf.placeholder(tf.float32)

probs = tf.nn.softmax(scores)# probability Scores of the predicted class
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, y))# Total cross-entropy loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(scores,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()#To save the model 
sess=tf.InteractiveSession()
sess.run(tf.initialize_all_variables())




k=0
batch_size = 20
m=1
L=1e-3#Learning Rate
flag = 0
G=20# Decay Learning rate by 10 folds after G epochs
for i in xrange(70000):
	sess.run(train_step,feed_dict={x:data_train[k:k+batch_size],y:labels_train[k:k+batch_size],keep_prob : 0.5,learning_rate:L,keep_prob_pool1:1.0,keep_prob_pool2:0.8})
	if(i%100 == 0):
		train_accuracy = accuracy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0,keep_prob_pool2:1.0})
		print("step %d, training accuracy%g\n"%(i, train_accuracy))
		loss = cross_entropy.eval(feed_dict={x:train_data[k:k+batch_size], y:train_labels[k:k+batch_size],keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0,keep_prob_pool2:1.0})
		print "loss is\n",loss
	k=k+batch_size
	if(k>=7000):
		test_accuracy = accuracy.eval(feed_dict={x:test_data, y:test_labels,keep_prob:1.0,learning_rate:L,keep_prob_pool1:1.0,keep_prob_pool2:1.0})
		print "epoch Finished, Test set accuracy is",float(test_accuracy),m
		k=0
		m=m+1
		flag = flag +1
		save_path = saver.save(sess, "/home/rishabh/Desktop/TensorFlow/CharNET MODEL/model.ckpt")#save Model after every epoch
  		print("Model saved in file: %s" % save_path)
	if(m%G == 0 and flag > 0):
		print "Decaying learning rate 10 Folds"
		L=L/10
		print L
		flag = 0
sess.close()
