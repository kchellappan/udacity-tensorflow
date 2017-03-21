#Deep Learning
#=============

#Assignment 6
#------------

#After training a skip-gram model in `5_word2vec.ipynb`, the goal 
#of this notebook is to train a LSTM character model over 
#[Text8](http://mattmahoney.net/dc/textdata) data.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

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

#filename = maybe_download('text8.zip', 31344016)
filename = 'text8.zip'

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))

#------------------------------------------------------------------

#Create a small validation set.

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

#------------------------------------------------------------------

#Utility functions to map characters to vocabulary IDs and back.

vocabulary_size = (len(string.ascii_lowercase) + 1) ** 2 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

#print(char2id('a'), char2id('z'), char2id(' '))
#print(id2char(1), id2char(26), id2char(0))

#Utility function to map bigrams to numeric IDs, with '  ' as ID 0
def bigram2id(bigram):
	firstCharID = char2id(bigram[0])
	secondCharID = char2id(bigram[1])
	bigramID = firstCharID*27 + secondCharID
	return bigramID
	
#Utility function to map numeric IDs back to bigrams
def id2bigram(dictid):
	firstChar = id2char(dictid // 27)
	secondChar = id2char(dictid % 27)
	bigram = firstChar + secondChar
	return bigram

#------------------------------------------------------------------

#Function to generate a training batch for the LSTM model.
batch_size=64
num_unrollings=10

#class BatchGenerator(object):
#  def __init__(self, text, batch_size, num_unrollings):
#    self._text = text
#    self._text_size = len(text)
#    self._batch_size = batch_size
#    self._num_unrollings = num_unrollings
#    segment = self._text_size // batch_size
#    self._cursor = [ offset * segment for offset in range(batch_size)]
#    self._last_batch = self._next_batch()
  
#  def _next_batch(self):
#    """Generate a single batch from the current cursor position in the data."""
#    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
#    for b in range(self._batch_size):
#      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
#      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
#    return batch
  
#  def next(self):
#    """Generate the next array of batches from the data. The array consists of
#    the last batch of the previous array, followed by num_unrollings new ones.
#    """
#    batches = [self._last_batch]
#    for step in range(self._num_unrollings):
#      batches.append(self._next_batch())
#    self._last_batch = batches[-1]
#    return batches

#batch_size is the number of parallel strings used (which form a 
#batch). Each string has a separate cursor.
class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size,1), dtype=np.int)
    for b in range(self._batch_size):
      bigramFirstChar = self._text[self._cursor[b]]
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
      
      bigramSecondChar = self._text[self._cursor[b]]
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
      
      bigramID = bigram2id(bigramFirstChar + bigramSecondChar)
      batch[b] = bigramID
      
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]
  
def bigrams(probabilities):
	return [id2bigram(b) for b in np.argmax(probabilities,1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

#print(batches2string(train_batches.next()))
#print(batches2string(train_batches.next()))
#print(batches2string(valid_batches.next()))
#print(batches2string(valid_batches.next()))

def logprob_unencodedlabels(predictions, labels):
	"""Log-probability of the true labels in a predicted batch."""
	predictions[predictions < 1e-10] = 1e-10
	
	one_hot_encoded_labels = np.zeros(shape=(labels.shape[0],vocabulary_size),dtype=np.float)
	for i in range(labels.shape[0]):
		one_hot_encoded_labels[i,labels[i]] = 1.0
	
	return np.sum(np.multiply(one_hot_encoded_labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

#------------------------------------------------------------------

#Defining the graph
num_nodes = 64 #Effectively, number of nodes of the hidden layer
	       #Size of the output and state of each LSTM cell
embedding_size = 128
num_sampled = 64
	       
graph_bigram = tf.Graph()
with graph_bigram.as_default():

	#We have 3 things to train on
	#1. The mappings from bigram ID space (equivalent to
	#   length 729 vectors) to embedded space (length
	#   embedding_size vectors)
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	
	#2. The weights of the input, forget, update and output 
	#   gates of the LSTM cell (This is effectively the weights
	#   of a hidden layer of num_nodes nodes)
	
	#Input gate: input, previous output, and bias.
	ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	ib = tf.Variable(tf.zeros([1, num_nodes]))

	#Forget gate: input, previous output, and bias.
	fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  	fb = tf.Variable(tf.zeros([1, num_nodes]))

 	#Memory cell: input, state and bias.                             
	cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	cb = tf.Variable(tf.zeros([1, num_nodes]))
	
	#Output gate: input, previous output, and bias.
	ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	ob = tf.Variable(tf.zeros([1, num_nodes]))
	
	#3. The weights of the final classifier (which is 
	#   effectively an output layer)
	w = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
	b = tf.Variable(tf.zeros([vocabulary_size]))
	
	#We also need to define variables to store the output and 
	#state from a previous session
	saved_output = tf.Variable(tf.zeros(shape=[batch_size, num_nodes]), trainable=False)
	saved_state = tf.Variable(tf.zeros(shape=[batch_size, num_nodes]), trainable=False)
	
	def lstm_cell(i, o, state):
		input_gate = tf.sigmoid(tf.matmul(i,ix) + tf.matmul(o,im) + ib)
		forget_gate = tf.sigmoid(tf.matmul(i,fx) + tf.matmul(o,fm) + fb)
		update = tf.tanh(tf.matmul(i,cx) + tf.matmul(o,cm) + cb)
		state = state*forget_gate + (input_gate * update)
		output_gate = tf.sigmoid(tf.matmul(i,ox) + tf.matmul(o,om) + ob) * tf.tanh(state)
		return output_gate, state
		
	#Create a list containing placeholders for all the input 
	#data
	train_data = list()
	for _ in range(num_unrollings+1):
		train_data.append(tf.placeholder(tf.int32,shape=[batch_size]))
	train_inputs = train_data[:num_unrollings]
	train_labels = train_data[1:]
	
	#Unroll the LSTM loop, and generate the outputs
	outputs = list()
	output = saved_output
	state = saved_state
	for i in train_inputs:
		embed_i = tf.nn.embedding_lookup(embeddings,i)
		output,state = lstm_cell(embed_i,output,state)
		outputs.append(output)

	#State saving across unrollings
	#I suspect tensorflow is like VHDL, it doesn't necessarily
	#execute statements sequentially, so we add the 
	#control_dependencies bit to force it to execute statements
	#in order (like a process in VHDL)
	with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
	
		#Take the outputs (of length num_nodes) of the LSTM cells
		#(effective hidden layers) and push them through the final
		#layer (classifier)
		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(w, b, tf.concat(0,outputs), tf.reshape(tf.concat(0,train_labels),[num_unrollings*batch_size,1]), num_sampled, vocabulary_size))
	
	#Define the optimizer
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(10.0,global_step,5000,0.1,staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	gradients,v = zip(*optimizer.compute_gradients(loss))
	gradients,_ = tf.clip_by_global_norm(gradients,1.25)
	optimizer = optimizer.apply_gradients(zip(gradients,v), global_step=global_step)
	
	#Make predictions on the data
	#These are very long vectors, so I'll have these lines commented,
	#to avoid slowing down the graph
	wt = tf.transpose(w)
	logits = tf.matmul(tf.concat(0,outputs),wt) + b
	train_prediction = tf.nn.softmax(logits)
	
	#Sampling and validation evaluation: batch 1, no unrolling
	sample_input = tf.placeholder(tf.int32, shape=[])
	embed_sample_input = tf.reshape(tf.nn.embedding_lookup(embeddings,sample_input),[1,128])
	saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
	saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
	
	#Grouping both operations required for resetting
	reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),saved_sample_state.assign(tf.zeros([1, num_nodes])))
	
	sample_output, sample_state = lstm_cell(embed_sample_input,saved_sample_output,saved_sample_state)
	with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
		sample_prediction = tf.nn.softmax(tf.matmul(sample_output, wt) + b)
		
num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph_bigram) as session:

	tf.initialize_all_variables().run()
	print('Initialized')
	
	mean_loss = 0
	for step in range(num_steps):
	
		batches = train_batches.next()
		feed_dict = dict()
		#Fill the list of input placeholders
		for i in range(num_unrollings+1):
			feed_dict[train_data[i]] = np.reshape(batches[i],[batch_size])
			
		_,l,predictions,lr = session.run([optimizer,loss,train_prediction,learning_rate],feed_dict=feed_dict)
		mean_loss = mean_loss + l
		
		#Print optimization summary
		if step%summary_frequency==0:
			
			if step>0:
				mean_loss = mean_loss/summary_frequency
				
			print('Average loss at step %d: %f learning rate: %f' % (step,mean_loss,lr))
			mean_loss = 0
			
			#predictions = train_prediction.eval()
			labels = np.concatenate(list(batches)[1:])
			print('Minibatch perplexity: %.2f' % float(np.exp(logprob_unencodedlabels(predictions, labels))))
			
			if step%(summary_frequency*10)==0:
			
				#Generate some sample output
				print('='*80)
				for i in range(5):
				
					feed = sample(random_distribution())
					feed = bigrams(feed)[0]
					sentence = feed
					reset_sample_state.run()
					
					for _ in range(79):
						prediction = sample_prediction.eval({sample_input: bigram2id(feed)})
						feed = sample(prediction)
						feed = bigrams(feed)[0]
						sentence += feed
					print(sentence)
				print('='*80)
				
			#Measure validation set perplexity
			reset_sample_state.run()
			valid_logprob = 0
			for _ in range(valid_size//2):
				b = valid_batches.next()
				predictions = sample_prediction.eval({sample_input: np.reshape(b[0],[])})
				valid_logprob = valid_logprob + logprob_unencodedlabels(predictions,b[1])
			
			print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / (valid_size/2))))
	
				

