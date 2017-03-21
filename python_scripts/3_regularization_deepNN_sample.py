import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

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

hidden_size=1024
hidden2_size=300

batch_size=128

beta1=0.01
beta2=0.01
beta3=0.0001

graph=tf.Graph()

with graph.as_default():
    
    tf_train_dataset=tf.placeholder(np.float32,shape=(batch_size,image_size*image_size))
    tf_train_labels=tf.placeholder(np.float32,shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)
    
    weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_size]))
    biases_1 = tf.Variable(tf.zeros([hidden_size]))
    
    hidden_1=tf.nn.relu(tf.matmul(tf_train_dataset,weights_1)+biases_1)
    
    #hidden_1=tf.nn.dropout(hidden_1,0.5)
    
    weights_2 = tf.Variable(
    tf.truncated_normal([hidden_size, hidden2_size]))
    biases_2 = tf.Variable(tf.zeros([hidden2_size]))
    
    hidden_2=tf.nn.relu(tf.matmul(hidden_1,weights_2)+biases_2)
    
    #hidden_2=tf.nn.dropout(hidden_2,0.5)
  
    weights_3 = tf.Variable(
    tf.truncated_normal([hidden2_size, num_labels]))
    biases_3 = tf.Variable(tf.zeros([num_labels]))
    
    logits=tf.matmul(hidden_2,weights_3)+biases_3
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))+beta1*tf.nn.l2_loss(weights_1)+beta2*tf.nn.l2_loss(weights_2)
    +beta3*tf.nn.l2_loss(weights_3)

    
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.04, global_step,1000,0.90,staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1),weights_2)+biases_2),weights_3)
    +biases_3)
    test_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1),weights_2)+biases_2),weights_3)
    +biases_3)
    
num_steps=9001

with tf.Session(graph=graph) as session:
    
    tf.initialize_all_variables().run()
    print('initialized')
    
    for step in range(num_steps):
        
        offset=(step*batch_size)%(train_dataset.shape[0]-batch_size)
        
        batch_data=train_dataset[offset:(offset+batch_size),:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        
        _,l,predictions=session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        
        if (step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
            
        global_step+=1
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
