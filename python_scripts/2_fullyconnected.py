#Assignment 2
#------------

#Previously in `1_notmnist.ipynb`, we created a pickle with formatted 
#datasets for training, development and testing on the [notMNIST dataset]
#(http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).

#The goal of this assignment is to progressively train deeper and more 
#accurate models using TensorFlow.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random

#-----------------------------------------------------------------------

#First reload the data we generated in `1_notmist.ipynb`.
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
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#-----------------------------------------------------------------------

#We're first going to train a multinomial logistic regression using 
#simple gradient descent.

#TensorFlow works like this:
#* First you describe the computation that you want to see performed: 
#  what the inputs, the variables, and the operations look like. These 
#  get created as nodes over a computation graph. This description is 
#  all contained within the block below:

#      with graph.as_default():
#          ...

#* Then you can run the operations on this graph as many times as you 
#  want by calling `session.run()`, providing it outputs to fetch from 
#  the graph that get returned. This runtime operation is all contained
#  in the block below:

#      with tf.Session(graph=graph) as session:
#          ...

#Let's load all the data into TensorFlow and build the computation graph
#corresponding to our training:

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random valued following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  
#-----------------------------------------------------------------------

#Let's run this computation and iterate:

num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
#-----------------------------------------------------------------------

#Let's now switch to stochastic gradient descent training instead, 
#which is much faster.

#The graph will be similar, except that instead of holding all the 
#training data into a constant node, we create a `Placeholder` node 
#which will be fed actual data at every call of `sesion.run()`.

batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)  
  
#-----------------------------------------------------------------------

#Let's run it:

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  
#-----------------------------------------------------------------------

#Problem
#-------

#Turn the logistic regression example with SGD into a 1-hidden layer 
#neural network with rectified linear units (nn.relu()) and 1024 hidden
#nodes. This model should improve your validation / test accuracy.

graphNN = tf.Graph() #Create a Tensorflow graph
no_of_hidden_layers = 1
no_of_hidden_nodes = 1024

#Augmenting training, validation and test sets
train_dataset = np.hstack([np.ones([train_dataset.shape[0],1]),train_dataset])
valid_dataset = np.hstack([np.ones([valid_dataset.shape[0],1]),valid_dataset])
test_dataset = np.hstack([np.ones([test_dataset.shape[0],1]),test_dataset])

with graphNN.as_default(): #Specify the default behavior of the graph as the problem statement (inputs, outputs, cost function definition, etc.)
    
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
    first_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size + 1, no_of_hidden_nodes],dtype=tf.float32))
    second_layer_weights = tf.Variable(tf.truncated_normal([no_of_hidden_nodes + 1, num_labels],dtype=tf.float32))

    #Define the outputs of the ML system. Here, the output is the output
    #layer, i.e, num_label softmax outputs, one for each class
    hidden_layer_output = tf.nn.relu(tf.matmul(tf_train_dataset, first_layer_weights))
    hidden_layer_output = tf.concat(1,[tf.ones([tf_train_dataset.get_shape()[0],1],dtype=tf.float32), hidden_layer_output])
    output_layer_output = (tf.matmul(hidden_layer_output, second_layer_weights))
    
    #Define the cost function of the neural network weight optimization
    #This is the mean cross entropy error (applying softmax to output layer)
    #betweem output and one-hot encoded labels
    loss = tf.cast(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_output, tf.cast(tf_train_labels,dtype=tf.float32))),dtype=tf.float32)
    
    #Define the optimizer (here, we'll use Stochastic Batch Gradient Descent,
    #which is Gradient Descent at its core, just with different training
    #sets at each iteration)
    #Learning rate is set to .5
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    #Generate the predicitions made by the network on the provided data
    train_prediction = tf.nn.softmax(output_layer_output)
    
    hidden_layer_output_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, first_layer_weights))
    hidden_layer_output_valid = tf.concat(1,[tf.ones([tf_valid_dataset.get_shape()[0],1]), hidden_layer_output_valid])
    output_layer_output_valid = (tf.matmul(hidden_layer_output_valid,second_layer_weights))
    valid_prediction = tf.nn.softmax(output_layer_output_valid)
    
    hidden_layer_output_test = tf.nn.relu(tf.matmul(tf_test_dataset, first_layer_weights))
    hidden_layer_output_test = tf.concat(1,[tf.ones([tf_test_dataset.get_shape()[0],1]), hidden_layer_output_test])
    output_layer_output_test = (tf.matmul(hidden_layer_output_test,second_layer_weights))
    test_prediction = tf.nn.softmax(output_layer_output_test)
                                          
num_iterations = 3001
                                          
#Describing a session for the optimization
with tf.Session(graph=graphNN) as session:
    
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
