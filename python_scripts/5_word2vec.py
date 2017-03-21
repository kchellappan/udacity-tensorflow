#Deep Learning
#=============

#Assignment 5
#------------

#The goal of this assignment is to train a skip-gram model over 
#[Text8](http://mattmahoney.net/dc/textdata) data.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

#-------------------------------------------------------------------

#Download the data from the source website if necessary.

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

#-------------------------------------------------------------------

#Read the data into a string.

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name)).split()
  f.close()
  
words = read_data(filename)
print('Data size %d' % len(words))

#-------------------------------------------------------------------

#Build the dictionary and replace rare words with UNK token.

vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

#-------------------------------------------------------------------

#Function to generate a training batch for the skip-gram model.

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    
#-------------------------------------------------------------------

#Train a skip-gram model.

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

#graph = tf.Graph()

#with graph.as_default():

  # Input data.
#  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
#  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
#  embeddings = tf.Variable(
#    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#  softmax_weights = tf.Variable(
#    tf.truncated_normal([vocabulary_size, embedding_size],
#                         stddev=1.0 / math.sqrt(embedding_size)))
#  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
#  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
#  loss = tf.reduce_mean(
#    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
#                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
#  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
#  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#  normalized_embeddings = embeddings / norm
#  valid_embeddings = tf.nn.embedding_lookup(
#    normalized_embeddings, valid_dataset)
#  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

#num_steps = 100001

#with tf.Session(graph=graph) as session:
#  tf.initialize_all_variables().run()
#  print('Initialized')
#  average_loss = 0
#  for step in range(num_steps):
#    batch_data, batch_labels = generate_batch(
#      batch_size, num_skips, skip_window)
#    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
#    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
#    average_loss += l
#    if step % 2000 == 0:
#      if step > 0:
#        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
#      print('Average loss at step %d: %f' % (step, average_loss))
#      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
#    if step % 10000 == 0:
#      sim = similarity.eval()
#      for i in xrange(valid_size):
#        valid_word = reverse_dictionary[valid_examples[i]]
#        top_k = 8 # number of nearest neighbors
#        nearest = (-sim[i, :]).argsort()[1:top_k+1]
#        log = 'Nearest to %s:' % valid_word
#        for k in xrange(top_k):
#          close_word = reverse_dictionary[nearest[k]]
#          log = '%s %s,' % (log, close_word)
#        print(log)
#  final_embeddings = normalized_embeddings.eval()
  
#-------------------------------------------------------------------

#num_points = 400

#tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

#def plot(embeddings, labels):
#  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
#  pylab.figure(figsize=(15,15))  # in inches
#  for i, label in enumerate(labels):
#    x, y = embeddings[i,:]
#    pylab.scatter(x, y)
#    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
#                   ha='right', va='bottom')
#  pylab.show()

#words = [reverse_dictionary[i] for i in range(1, num_points+1)]
#plot(two_d_embeddings, words)

#-------------------------------------------------------------------

#Problem
#-------

#An alternative to Word2Vec is called [CBOW]
#(http://arxiv.org/abs/1301.3781) (Continuous Bag of Words). In 
#the CBOW model, instead of predicting a context word from a word 
#vector, you predict a word from the sum of all the word vectors 
#in its context. Implement and evaluate a CBOW model trained on 
#the text8 dataset.

#First, a function to generate batches of data to work with.
#num_skips is how many times you reuse a word (as input for 
#word2vec, target for CBOW). For CBOW, should be only once, so
#num_skips = 1. Leaving it as a free variable for now.
data_index=0
def generate_CBOW_batch(batch_size,num_skips,skip_window):
	global data_index
	span = 2*skip_window+1
	batch = np.ndarray(shape=(batch_size,1), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size*(span-1)), dtype=np.int32)
	buffer = collections.deque(maxlen=span)
	
	#Initializing the buffer
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1)% len(data)
		
	for i in range(batch_size//num_skips):
		target = skip_window
		input = list()
		
		for j in range(len(buffer)):
		
			if (j==target):
				continue
			
			input.append(buffer[j])
			
		batch[i] = buffer[skip_window]
		labels[((span-1)*i):((span-1)*(i+1))] = np.array(input)
		
		buffer.append(data[data_index])
		data_index = (data_index + 1)%len(data)
		
	return np.array(batch), labels
	
#Building the graph
batch_size=128
num_skips=1
skip_window=3
valid_size=16
valid_window=100

valid_examples = np.array(random.sample(range(valid_window),valid_size))
num_sampled = 64
embedding_size = 128

avg_matrix = np.zeros(shape=(batch_size,batch_size*2*skip_window),dtype=np.float32)
for i in range(batch_size):
	avg_matrix[i,(2*skip_window*i):(2*skip_window*(i+1))] = 1.0/2/skip_window


graph_CBOW = tf.Graph()
with graph_CBOW.as_default():

	#tf_train_labels = tf.placeholder(shape=[batch_size,1],dtype=tf.int32)
	tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	tf_train_dataset = tf.placeholder(shape=[batch_size*2*skip_window],dtype=tf.int32)
	#tf_avg_matrix = tf.constant(avg_matrix,dtype=tf.int32)
	
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
	softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
	softmax_bias = tf.Variable(tf.zeros([vocabulary_size]))
	
	embed = tf.nn.embedding_lookup(embeddings,tf_train_dataset)
	embed = tf.matmul(avg_matrix,embed)
	 
	loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights,softmax_bias,embed,tf_train_labels,num_sampled,vocabulary_size))
	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
	
num_steps = 100001
num_examples = 20

with tf.Session(graph=graph_CBOW) as session:

	tf.initialize_all_variables().run()
	print('Initialized')
	average_loss=0
	
	for step in range(num_steps):
		
		batch_labels, batch_data = generate_CBOW_batch(batch_size,num_skips,skip_window)
		feed_dict = {tf_train_labels: batch_labels, tf_train_dataset: batch_data}
		_,l,e,sw,sb = session.run([optimizer,loss,embeddings,softmax_weights,softmax_bias],feed_dict=feed_dict)
		average_loss+=l
		
		if step%2000==0:
		
			if step>0:
				
				average_loss = average_loss/2000
				print('Average loss at step %d is %f' % (step, average_loss))
				average_loss=0
	
	print('')
	print('Couple of examples to see if CBOW is working as it should...')
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(e), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	avg_matrix = np.ones(shape=(1,6),dtype=np.float32)*1.0/6
	for i in range(num_examples):
	
		data_index = (data_index + 100)%len(data)
		example_labels, example_data = generate_CBOW_batch(1,num_skips,skip_window)
		
		example_embeddings = tf.nn.embedding_lookup(e,example_data)
		example_embeddings = tf.matmul(avg_matrix,example_embeddings)
		similarity = tf.matmul(example_embeddings, tf.transpose(normalized_embeddings))
		sim = similarity.eval()
		
		top_k = 8 # number of nearest neighbors
		nearest = (-sim[0, :]).argsort()[0:top_k]
		log=''
		for k in xrange(top_k):
			close_word = reverse_dictionary[nearest[k]]
			log = '%s %s,' % (log, close_word)
		
		print('')
		print('Example %d @ data_index %d:' % (i,data_index))
		print('    context:', [reverse_dictionary[bi] for bi in example_data])
		print('    labels:', [reverse_dictionary[li] for li in example_labels[0,:]])
		print('	   best guesses:', log)
		
