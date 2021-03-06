#Assignment 4
#------------

#Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we
#trained fully connected networks to classify [notMNIST]
#(http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) 
#characters.

#The goal of this assignment is make the neural network convolutional.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random

#-----------------------------------------------------------------------

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
#-----------------------------------------------------------------------

#Reformat into a TensorFlow-friendly shape:
#- convolutions need the image data formatted as a cube (width by height
#  by #channels)
#- labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#-----------------------------------------------------------------------

#Let's build a small network with two convolutional layers, followed by 
#one fully connected layer. Convolutional networks are more expensive 
#computationally, so we'll limit its depth and number of fully connected
#nodes.

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

#graph = tf.Graph()

#with graph.as_default():

  # Input data.
#  tf_train_dataset = tf.placeholder(
#    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#  tf_valid_dataset = tf.constant(valid_dataset)
#  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
#  layer1_weights = tf.Variable(tf.truncated_normal(
#      [patch_size, patch_size, num_channels, depth], stddev=0.1))
#  layer1_biases = tf.Variable(tf.zeros([depth]))
#  layer2_weights = tf.Variable(tf.truncated_normal(
#      [patch_size, patch_size, depth, depth], stddev=0.1))
#  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#  layer3_weights = tf.Variable(tf.truncated_normal(
#      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#  layer4_weights = tf.Variable(tf.truncated_normal(
#      [num_hidden, num_labels], stddev=0.1))
#  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
#  def model(data):
#    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer1_biases)
#    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer2_biases)
#    shape = hidden.get_shape().as_list()
#    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#    return tf.matmul(hidden, layer4_weights) + layer4_biases
 
  # Training computation.
#  logits = model(tf_train_dataset)
#  loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
#  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
#  train_prediction = tf.nn.softmax(logits)
#  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#  test_prediction = tf.nn.softmax(model(tf_test_dataset))
#  num_steps = 1001

#with tf.Session(graph=graph) as session:
#  tf.initialize_all_variables().run()
#  print('Initialized')
#  for step in range(num_steps):
#    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#    batch_labels = train_labels[offset:(offset + batch_size), :]
#    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#    _, l, predictions = session.run(
#      [optimizer, loss, train_prediction], feed_dict=feed_dict)
#    if (step % 50 == 0):
#      print('Minibatch loss at step %d: %f' % (step, l))
#      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#      print('Validation accuracy: %.1f%%' % accuracy(
#        valid_prediction.eval(), valid_labels))
#  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

#-----------------------------------------------------------------------

#Problem 1
#---------

#The convolutional model above uses convolutions with stride 2 to 
#reduce the dimensionality. Replace the strides by a max pooling 
#operation (`nn.max_pool()`) of stride 2 and kernel size 2.

graph_convNN = tf.Graph()

with graph_convNN.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
	
	tf_test_dataset = tf.constant(test_dataset, tf.float32)
	tf_valid_dataset = tf.constant(valid_dataset, tf.float32)
	
	first_layer_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
	first_layer_bias = tf.Variable(tf.zeros([depth]))
	
	second_layer_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	second_layer_bias = tf.Variable(tf.constant(1.0, shape=[depth]))
	
	third_layer_weights = tf.Variable(tf.truncated_normal([image_size // 4  * image_size // 4 * depth, num_hidden], stddev=0.1))
	third_layer_bias = tf.Variable(tf.constant(1.0,shape=[num_hidden]))
	
	fourth_layer_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels],stddev=0.1))
	fourth_layer_bias = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
	#A function is defined to perform the convolution network operations
	def model(data):
	
		#First layer operation
		conv = tf.nn.relu(tf.nn.conv2d(data,first_layer_weights, [1, 1, 1, 1], padding='SAME') + first_layer_bias)
		pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],'SAME')
		
		#Second layer operation
		conv = tf.nn.relu(tf.nn.conv2d(pool,second_layer_weights,[1,1,1,1],'SAME') + second_layer_bias)
		pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],'SAME')
		
		#Third layer operation
		shape = pool.get_shape().as_list()
		reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape,third_layer_weights) + third_layer_bias)
		
		#Fourth (final) layer operation
		hidden = tf.nn.relu(tf.matmul(hidden, fourth_layer_weights) + fourth_layer_bias)
		
		return hidden
		
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))
	
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model(tf_test_dataset))
	valid_prediction = tf.nn.softmax(model(valid_dataset))
	
num_steps = 1001
with tf.Session(graph=graph_convNN) as session:

	tf.initialize_all_variables().run()
	print('Initialized')
	
	for step in range(num_steps):
	
		training_samples_used = random.sample(range(train_dataset.shape[0]),batch_size)
		batch_data = train_dataset[training_samples_used,:,:,:]
		batch_labels = train_labels[training_samples_used,:]
		
		feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
		
		_,l,predictions = session.run([optimizer,loss,train_prediction], feed_dict=feed_dict)
		
		if (step%50==0):
			print('Minibatch loss at step %d: %f' % (step,l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions,batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(),valid_labels))
			
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),test_labels))

#-----------------------------------------------------------------------

#Problem 2
#---------

#Try to get the best performance you can using a convolutional net. Look
#for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/)
#architecture, adding Dropout, and/or adding learning rate decay.

first_layer_depth = 4
second_layer_depth = 6
first_layer_patch_size = 5
second_layer_patch_size = 5
num_hidden = 500
num_labels = 10

graph_LeNet5 = tf.Graph()

with graph_LeNet5.as_default():

	tf_train_dataset = tf.placeholder(shape=(batch_size,image_size,image_size,num_channels), dtype=tf.float32)
	tf_train_labels = tf.placeholder(shape=(batch_size,num_labels),dtype=tf.float32)
	tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.floar32)
	tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)
	
	first_layer_weights = tf.Variable(tf.truncated_normal([first_layer_patch_size,first_layer_patch_size,num_channels,first_layer_depth],stddev=0.1))
	first_layer_bias = tf.Variable(tf.constant(1.0,shape=[first_layer_depth]))
	
	second_layer_weights = tf.Variable(tf.truncated_normal([second_layer_patch_size,second_layer_patch_size,first_layer_depth,second_layer_depth],stddev=0.1))
	second_layer_bias = tf.Variable(tf.constant(1.0,shape=[second_layer_depth]))
	
	third_layer_weights = tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * second_layer_depth, num_hidden],stddev=0.1))
	third_layer_bias = tf.Variable(tf.constant(1.0,shape=[num_hidden]))
	
	fourth_layer_weights = tf.Vraiable(tf.truncated_normal([num_hidden, num_labels],stddev=0.1))
	fourth_layer_weights = tf.Variable(tf.constant(1.0,shape=[num_labels]))
	
	def model(data):
	
		first_layer_op = tf.nn.conv2d(tf_train_dataset,first_layer_weights,[1,1,1,1],padding='SAME') + first_layer_bias
		first_layer_op = tf.nn.relu(tf.nn.max_pool(first_layer_op,[1,2,2,1],[1,2,2,1],padding='SAME'))
		
		second_layer_op = tf.nn.conv2d(first_layer_op,second_layer_weights,[1,1,1,1],padding='SAME') + second_layer_bias
		second_layer_op = tf.nn.relu(tf.nn.max_pool(second_layer_op,[1,2,2,1],[1,2,2,1],padding='SAME'))
		
		data_shape = second_layer_op.get_shape().as_list()
		reshaped_data = reshape(data_shape,[data_shape[0], data_shape[1]*data_shape[2]*data_shape[3]])
		
		third_layer_op = tf.nn.relu(tf.matmul(reshaped_data,third_layer_weights) + third_layer_bias)
		fourth_layer_op = tf.nn.relu(tf.matmul(third_layer_op,fourth_layer_weights) + fourth_layer_bias)
		
		global_step = tf.Variable(0)
		
		return fourth_layer_op
		
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))
	
	learn_rate = tf.train.exponential_decay(0.04,global_step,1000,0.9, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
	
	train_predictions = tf.nn.softmax(logits)
	test_predictions = tf.nn.softmax(model(test_dataset))
	valid_predictions = tf.nn.softmax(model(valid_dataset))
	
num_iterations = 1001

with tf.Session(graph=graph_LeNet5) as session:

	tf.initialize_all_variables().run()
	print('Initialized')

	for step in range(num_iterations):
		
		training_samples_used = random.sample(range(train_dataset.shape[0]),batch_size)
		batch_dataset = train_dataset[training_samples,:,:,:]
		batch_dataset = train_labels[training_samples,:]
	
		feed_dict={tf_train_dataset: batch_dataset, tf_train_labels: batch_labels}
		_,l,predictions = session.run([optimizer,loss,train_predictions], feed_dict=feed_dict)
		
		global_step+=1
		
		if (step%50==0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
			
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
