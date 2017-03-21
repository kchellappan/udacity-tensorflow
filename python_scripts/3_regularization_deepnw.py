#Deep Learning
#=============

#Assignment 3
#------------

#Previously in `2_fullyconnected.ipynb`, you trained a logistic 
#regression and a neural network model.

#The goal of this assignment is to explore regularization techniques.

#These are all the modules we'll be using later. Make sure you can 
#import them before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import random

#-----------------------------------------------------------------------

#First reload the data we generated in _notmist.ipynb_.

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

#Reformat into a shape that's more adapted to the models we're going to
#train:
#- data as a flat matrix,
#- labels as float 1-hot encodings.

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


batch_size = 128
no_of_hidden_nodes = 1024
reg_parameter = 0.01;
num_iterations = 3001
dropout_prob = 0.75

train_dataset = np.hstack([np.ones([train_dataset.shape[0],1]),train_dataset])
valid_dataset = np.hstack([np.ones([valid_dataset.shape[0],1]),valid_dataset])
test_dataset = np.hstack([np.ones([test_dataset.shape[0],1]),test_dataset])

graphNN_dropout = tf.Graph()

with graphNN_dropout.as_default(): #Specify the default behavior of the graph as the problem statement (inputs, outputs, cost function definition, etc.)
    
    #Create placeholdera to contain the training set (data and labels)
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size + 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
    
    #Define the validation set as a constant (it doesn't change throughout the optimization anyway)
    tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
    
    #Similarly, define the test set as constant
    tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)
    
    #Define the variables of the optimization.
    #In this case, they will be the weights from the input to hidden layer
    #(1024 rows and 28*28 columns) and the weights from the hidden to
    #output layer (num_labels rows and 1024 columns), with biases
    #for each layer (1024 and num_labels length vectors respectively)
    #Initializing them all with truncated normal RV values
    first_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size + 1, 1024],dtype=tf.float32))
    second_layer_weights = tf.Variable(tf.truncated_normal([1024 + 1, 500],dtype=tf.float32))
    third_layer_weights = tf.Variable(tf.truncated_normal([500 + 1, 30],dtype=tf.float32))
    fourth_layer_weights = tf.Variable(tf.truncated_normal([30 + 1, num_labels],dtype=tf.float32))

    #Define the outputs of the ML system. Here, the output is the output
    #layer, i.e, num_label softmax outputs, one for each class
    hidden_layer_output1 = tf.nn.relu(tf.matmul(tf_train_dataset, first_layer_weights))
    #hidden_layer_output1 = tf.nn.dropout(hidden_layer_output1,dropout_prob)
    hidden_layer_output1 = tf.concat(1,[tf.ones([tf_train_dataset.get_shape()[0],1],dtype=tf.float32), hidden_layer_output1])
    
    hidden_layer_output2 = tf.nn.relu(tf.matmul(hidden_layer_output1, second_layer_weights))
    #hidden_layer_output2 = tf.nn.dropout(hidden_layer_output2,dropout_prob)
    hidden_layer_output2 = tf.concat(1,[tf.ones([tf_train_dataset.get_shape()[0],1],dtype=tf.float32), hidden_layer_output2])
    
    hidden_layer_output3 = tf.nn.relu(tf.matmul(hidden_layer_output2, third_layer_weights))
    #hidden_layer_output3 = tf.nn.dropout(hidden_layer_output3,dropout_prob)
    hidden_layer_output3 = tf.concat(1,[tf.ones([tf_train_dataset.get_shape()[0],1],dtype=tf.float32), hidden_layer_output3])
    
    output_layer_output = (tf.matmul(hidden_layer_output3, fourth_layer_weights))
    
    #Define the cost function of the neural network weight optimization
    #This is the mean cross entropy error (applying softmax to output layer)    
    #between output and one-hot encoded labels
    #loss = tf.cast(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_output, tf.cast(tf_train_labels,dtype=tf.float32))) + reg_parameter * (tf.nn.l2_loss(first_layer_weights) + tf.nn.l2_loss(second_layer_weights) + tf.nn.l2_loss(third_layer_weights) + tf.nn.l2_loss(fourth_layer_weights)),dtype=tf.float32)
    output_layer_output = tf.nn.softmax(output_layer_output)
    loss = tf.cast(tf.reduce_mean(-1 * tf.mul(tf_train_labels,tf.log(output_layer_output + 1e-9))),dtype=tf.float32)
    
    #Define the optimizer (here, we'll use Stochastic Batch Gradient Descent,
    #which is Gradient Descent at its core, just with different training
    #sets at each iteration)
    #Learning rate is set to .5
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Generate the predicitions made by the network on the provided data
    train_prediction = output_layer_output
    
    hidden_layer_output_valid1 = tf.nn.relu(tf.matmul(tf_valid_dataset, first_layer_weights))
    hidden_layer_output_valid1 = tf.concat(1,[tf.ones([tf_valid_dataset.get_shape()[0],1]), hidden_layer_output_valid1])
    
    hidden_layer_output_valid2 = tf.nn.relu(tf.matmul(hidden_layer_output_valid1, second_layer_weights))
    hidden_layer_output_valid2 = tf.concat(1,[tf.ones([tf_valid_dataset.get_shape()[0],1]), hidden_layer_output_valid2])
    
    hidden_layer_output_valid3 = tf.nn.relu(tf.matmul(hidden_layer_output_valid2, third_layer_weights))
    hidden_layer_output_valid3 = tf.concat(1,[tf.ones([tf_valid_dataset.get_shape()[0],1]), hidden_layer_output_valid3])
    
    output_layer_output_valid = (tf.matmul(hidden_layer_output_valid3,fourth_layer_weights))
    valid_prediction = tf.nn.softmax(output_layer_output_valid)
    
    hidden_layer_output_test1 = tf.nn.relu(tf.matmul(tf_test_dataset, first_layer_weights))
    hidden_layer_output_test1 = tf.concat(1,[tf.ones([tf_test_dataset.get_shape()[0],1]), hidden_layer_output_test1])
    
    hidden_layer_output_test2 = tf.nn.relu(tf.matmul(hidden_layer_output_test1, second_layer_weights))
    hidden_layer_output_test2 = tf.concat(1,[tf.ones([tf_test_dataset.get_shape()[0],1]), hidden_layer_output_test2])
    
    hidden_layer_output_test3 = tf.nn.relu(tf.matmul(hidden_layer_output_test2, third_layer_weights))
    hidden_layer_output_test3 = tf.concat(1,[tf.ones([tf_test_dataset.get_shape()[0],1]), hidden_layer_output_test3])
    
    output_layer_output_test = (tf.matmul(hidden_layer_output_test3,fourth_layer_weights))
    test_prediction = tf.nn.softmax(output_layer_output_test)
                                          
num_iterations = 3001
                                          
#Describing a session for the optimization
with tf.Session(graph=graphNN_dropout) as session:
    
    #Initialize all the variables described in the graph
    tf.initialize_all_variables().run()
    
    #Iterating for num_iterations times
    for step in range(num_iterations):

        #Pull out some random training examples to form 
        #a batch for this iterations
        indices_used = random.sample(range(train_dataset.shape[0]),batch_size)
        batch_data = train_dataset[indices_used,:]
        batch_labels = train_labels[indices_used,:]
                                          
        #Define the feed_dict: A dictionary telling the graph how to
        #populate the placeholders
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        #When you run a session, you provide a tuple of arguments
        #If the ith element of the tuple is an operation (optimizer),
        #the ith return element is None
        #For any variables passed in, the corresponding return element
        #is the object defined inside the graph with the same name
        _,l,predictions = session.run([optimizer,loss,train_prediction], feed_dict = feed_dict)

        #Print reports after so many iterations
        if (step%500 == 0):
            print("Minibatch loss at step %d: %f" % (step,l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            
            #I think you can choose to selectively evaluate some quantities
            #as opposed to having to evaluate at every iterations
            #and lose some CPU cycles
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
