import config as cfg
import tensorflow as tf
import itchat
from tqdm import tqdm

class Solver():
	def __init__(self, model, data, wechat = False):
		self.model = model
		self.data = data

		self.max_iter = cfg.MAX_ITER
		self.summary_iter = cfg.SUMMARY_ITER
		self.print_iter = cfg.PRINT_ITER

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
		
		self.output_dir = cfg.OUTPUT_DIR
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.output_dir)

		self.wechat = wechat
		if wechat:
			itchat.auto_login(enableCmdQR=True)

	def train(self, init = True):
		with tf.Session() as sess:
			if init:
				sess.run(tf.global_variables_initializer())
				print("INIT DONE")
			for i in tqdm(range(self.max_iter)):
				X_train, y_train = self.data.get()
				feed_dict = {self.model.X: X_train, self.model.y_hat: y_train}
				
				if (i % self.summary_iter != 0):
					loss, loss_coor, loss_confidence, loss_classes, _ = sess.run([self.model.loss,
						self.model.loss_coor,
						self.model.loss_confidence,
						self.model.loss_classes,
						self.train_op], feed_dict = feed_dict)
				else:
					loss, loss_coor, loss_confidence, loss_classes, summary, _ = sess.run([self.model.loss,
						self.model.loss_coor,
						self.model.loss_confidence,
						self.model.loss_classes,
						self.summary_op,
						self.train_op], feed_dict = feed_dict)
					self.writer.add_summary(summary, i)

				if (i % self.print_iter == 0):
					print("Iter#", i)
					print("learning_rate:", self.learning_rate.eval())
					print("loss:", loss)
					print("loss_coor: ", loss_coor)
					print("loss_confidence: ", loss_confidence)
					print("loss_classes: ", loss_classes)

					if self.wechat:
						itchat.send('Iter#{}\n'.format(i)
									+ 'learning_rate: {}\n'.format(self.learning_rate.eval())
									+ 'loss: {}\n'.format(loss)
									+ 'loss_coor: {}\n'.format(loss_coor)
									+ 'loss_confidence: {}\n'.format(loss_confidence)
									+ 'loss_classes: {}\n'.format(loss_classes), toUserName='filehelper')
			if self.wechat:
				itchat.send('DONE TRAINING', toUserName='filehelper')