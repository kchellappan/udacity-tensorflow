from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import random

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
  
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
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

num_iterations = 9001
batch_size = 1000
l2_reg_parameter = 0.1
dropout_prob = 1
num_nodes_layer_1 = 1024
num_nodes_layer_2 = 500
num_nodes_layer_3 = 30
print(train_dataset.shape)

graph = tf.Graph()
with graph.as_default():

	tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size*image_size))
	tf_train_labels = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_labels))
	
	tf_test_dataset = tf.constant(test_dataset)
	
	tf_valid_dataset = tf.constant(valid_dataset)
	
	first_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_nodes_layer_1],dtype=tf.float32))
	first_layer_bias = tf.Variable(tf.truncated_normal([1,num_nodes_layer_1],dtype=tf.float32))
	second_layer_weights = tf.Variable(tf.truncated_normal([num_nodes_layer_1,num_nodes_layer_2],dtype=tf.float32))
	second_layer_bias = tf.Variable(tf.truncated_normal([1,num_nodes_layer_2],dtype=tf.float32))
	third_layer_weights = tf.Variable(tf.truncated_normal([num_nodes_layer_2,num_labels],dtype=tf.float32))
	third_layer_bias = tf.Variable(tf.truncated_normal([1,num_labels],dtype=tf.float32))
	
	#tf_train_dataset = tf.nn.dropout(tf_train_dataset,dropout_prob)
	first_layer_output = tf.nn.tanh(tf.matmul(tf_train_dataset,first_layer_weights)) + first_layer_bias
	second_layer_output = tf.nn.tanh(tf.matmul(first_layer_output,second_layer_weights)) + second_layer_bias
	third_layer_output = tf.matmul(second_layer_output,third_layer_weights) + third_layer_bias
	
	#cross_entropy = -tf.reduce_sum(tf_train_labels * tf.log(third_layer_output + 1e-9))
	#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(third_layer_output,tf_train_labels)) + l2_reg_parameter * (tf.nn.l2_loss(first_layer_weights) + tf.nn.l2_loss(second_layer_weights) + tf.nn.l2_loss(third_layer_weights))
	global_step = tf.Variable(0)
	learn_rate = tf.train.exponential_decay(0.04,global_step,1000,0.90,staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)
	train_prediction = tf.nn.softmax(third_layer_output)
	
	first_layer_output_test = tf.nn.tanh(tf.matmul(tf_test_dataset,first_layer_weights) + first_layer_bias)
	second_layer_output_test = tf.nn.tanh(tf.matmul(first_layer_output_test,second_layer_weights) + second_layer_bias)
	third_layer_output_test = tf.nn.softmax(tf.matmul(second_layer_output_test,third_layer_weights) + third_layer_bias)
	test_prediction = third_layer_output_test
	
	first_layer_output_valid = tf.nn.tanh(tf.matmul(tf_valid_dataset,first_layer_weights) + first_layer_bias)
	second_layer_output_valid = tf.nn.tanh(tf.matmul(first_layer_output_valid,second_layer_weights) + second_layer_bias)
	third_layer_output_valid = tf.nn.softmax(tf.matmul(second_layer_output_valid,third_layer_weights) + third_layer_bias)
	valid_prediction = third_layer_output_valid
	
with tf.Session(graph=graph) as session:

	tf.initialize_all_variables().run()
	
	for step in range(num_iterations):
	
		indices_used = random.sample(range(train_dataset.shape[0]),batch_size)
		batch_dataset = train_dataset[indices_used,:]
		batch_labels = train_labels[indices_used,:]
		feed_dict = {tf_train_dataset: batch_dataset, tf_train_labels: batch_labels}
		
		_,l,predictions = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
		
		if (step%500==0):
			print("Minibatch loss at step %d: %f" % (step,l))
            		print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
	        	print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
	        	
	        
	        global_step = global_step + 1

	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
