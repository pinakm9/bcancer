import tensorflow as tf
import math
from paths import *

class NeuralNet:
	
	def __init__(self, feature_count, label_count, learning_rate = 0.00001, momentum = 0.4, hnodes = [150]):
		self.x = tf.placeholder(tf.float32, [None, feature_count])
		# Declare the output data placeholder
		self.y = tf.placeholder(tf.float32, [None, label_count])
		# Declare the weights connecting the input to the hidden layer 1
		self.W1 = tf.Variable(tf.random_normal([feature_count, hnodes[0]], stddev=0.03), name='W1')
		self.b1 = tf.Variable(tf.random_normal([hnodes[0]]), name='b1')
		# Weights connecting the hidden layer 1 to the output layer
		self.W2 = tf.Variable(tf.random_normal([hnodes[0], label_count], stddev=0.04), name='W2')
		self.b2 = tf.Variable(tf.random_normal([label_count]), name='b2')
		# Calculate the output of the hidden layers
		self.hidden_out1 = tf.add(tf.matmul(self.x, self.W1), self.b1)
		self.hidden_out1 = tf.nn.relu(self.hidden_out1)
		self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.hidden_out1, self.W2), self.b2))
		# Calculate softmax activated output layer	
		self.y_c = tf.clip_by_value(self.y_, 1e-10, 0.9999999)
		self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.y*tf.log(self.y_c) + (1-self.y)*tf.log(1-self.y_c), axis=1))
		# Add an optimizer
		self.optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum).minimize(self.cross_entropy)
		# Setup the initialization operator
		self.init_op = tf.global_variables_initializer()
		# Add ops to save and restore all the variables
		self.saver = tf.train.Saver()
		# Define an accuracy assessment operation
		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.hnodes = hnodes

	def train(self, data, epochs = 500, batch_count = 3):
		# Start the session
		cost0, fails = 1e9, 0
		with tf.Session() as sess:
		# Initialize the variables
			sess.run(self.init_op)
			batch_size = math.ceil(data.train.count/float(batch_count))
			for epoch in range(epochs):
				avg_cost = 0
				for i in range(batch_count):
					batch = data.train.next(batch_size)
					_, c = sess.run([self.optimizer, self.cross_entropy], feed_dict={self.x: batch.features, self.y: batch.labels})
					avg_cost += c / batch_count
				# Fail-safe if the minimization stagnates
				if cost0 < avg_cost and avg_cost > 1e-6:
					fails += 1
					if fails == 50:
						print("Method strayed away from minima")
						return False # Training was unsuccessful
				else:
					cost0 = avg_cost
					fails = 0
				print("Epoch: {}, cost = {:.9f}".format((epoch + 1), avg_cost))
			self.saver.save(sess, p2_nn_model)
			print('Training completed.')
			self.epochs = epochs
		return True # Training was successful

	def test(self, data):
		with tf.Session() as sess:
			self.saver.restore(sess, p2_nn_model)
			train_acc = sess.run(self.accuracy, feed_dict={self.x: data.train.features, self.y: data.train.labels})
			# Print results
			test_acc = sess.run(self.accuracy, feed_dict={self.x: data.test.features, self.y: data.test.labels})
			print('Accuracy on training data: {:.2f}%'.format(100*train_acc))
			print('Accuracy on test data: {:.2f}%'.format(100*test_acc))
		# Store the results in a text file
		with open(p2_experiment, 'a+') as file:
			file.write('{}  {:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{}\n'\
				.format(self.epochs, self.learning_rate, self.momentum, train_acc*100, test_acc*100, self.hnodes))
		