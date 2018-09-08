import config as cfg
import tensorflow as tf


class Solver():
	def __init__(self, model, data):
		self.model = model
		self.data = data
		self.initial_learning_rate = cfg.LEARNING_RATE
		self.decay_rate = cfg.DECAY_RATE
		self.decay_steps = cfg.DECAY_STEPS
		self.staircase = cfg.STAIRCASE
		self.global_step = tf.train.create_global_step()
		self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
			self.global_step,
			self.decay_steps,
			self.decay_rate,
			self.staircase,
			name='learning_rate')
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		self.train_op = self.optimizer.minimize(self.model.loss, global_step = self.global_step)
		self.summary_op = tf.summary.merge_all()

	def train(self, num_iter, summary_iter, init = True):
		with tf.Session() as sess:
			if init:
				
			for i in range(num_iter):
				X_train, y_train = self.data.get()
				feed_dict = {"X": X_train, "y_hat": y_train}
				if (i % summary_iter == 0):
					loss, _ = sess.run([self.model.loss, self.train_op], feed_dict = feed_dict)
				else:
					loss，summary, _ = sess.run([self.model.loss, self.summary_op, self.train_op], feed_dict = feed_dict)
					self.writer.add_summary(summary, i)