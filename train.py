import config as cfg
import tensorflow as tf
import itchat
import datetime
import os
from tqdm import tqdm

class Solver():
	def __init__(self, model, data, wechat = False):
		self.model = model
		self.data = data

		self.max_iter = cfg.MAX_ITER
		self.summary_iter = cfg.SUMMARY_ITER
		self.print_iter = cfg.PRINT_ITER

		self.learning_rate = cfg.LEARNING_RATE
		#self.decay_rate = cfg.DECAY_RATE
		#self.decay_steps = cfg.DECAY_STEPS
		#self.staircase = cfg.STAIRCASE
		self.global_step = tf.train.create_global_step()
		#self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
		#	self.global_step,
		#	self.decay_steps,
		#	self.decay_rate,
		#	self.staircase,
		#	name='learning_rate')

		self.epsilon = cfg.EPSILON
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon = self.epsilon)
		grads, variables = zip(*self.optimizer.compute_gradients(self.model.loss))
		grads, global_norm = tf.clip_by_global_norm(grads, 5)
		self.train_op = self.optimizer.apply_gradients(zip(grads, variables))
		
		self.output_dir = cfg.OUTPUT_DIR
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.output_dir)

		print('fuck', datetime.datetime.now().strftime('%m-%d %H:%M:%S'))
		self.save_iter = cfg.SAVE_ITER
		self.save_dir = os.path.join(cfg.SAVE_DIR, datetime.datetime.now().strftime('%m-%d_%H:%M:%S'))
		os.makedirs(self.save_dir)
		self.save_cfg()
		self.ckpt_file = os.path.join(self.save_dir, 'yolo.ckpt')
		self.saver = tf.train.Saver()

		self.wechat = wechat
		if wechat:
			itchat.auto_login(enableCmdQR=True)

	def train(self, init = True):
		with tf.Session() as sess:
			if init:
				sess.run(tf.global_variables_initializer())
				print("INIT DONE")
			else:
				self.weights_file = os.path.join(cfg.WEIGHTS_DIR, cfg.WEIGHTS_FILE)
				print('Restoring weights from: ' + self.weights_file)
				self.saver.restore(sess, self.weights_file)

			current_epoch = 0

			for i in tqdm(range(self.max_iter)):
				X_train, y_train = self.data.get()
				feed_dict = {self.model.X: X_train, self.model.y_hat: y_train}
				
				if (self.summary_iter > 0) and (i % self.summary_iter != 0):
					loss, loss_coor, loss_iou, loss_classes, _ = sess.run([self.model.loss,
						self.model.loss_coor,
						self.model.loss_iou,
						self.model.loss_classes,
						self.train_op], feed_dict = feed_dict)
				else:
					loss, loss_coor, loss_iou, loss_classes, summary, _ = sess.run([self.model.loss,
						self.model.loss_coor,
						self.model.loss_iou,
						self.model.loss_classes,
						self.summary_op,
						self.train_op], feed_dict = feed_dict)
					self.writer.add_summary(summary, i)

				if (self.save_iter <= 0 ) or (i % self.save_iter == 0):
					print('{} Saving checkpoint file to: {}'.format(
						datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
						self.save_dir))
					self.saver.save(sess, self.ckpt_file, global_step=self.global_step)

				text = 'loss: {}\n'.format(loss) + 'loss_coor: {}\n'.format(loss_coor) + 'loss_iou: {}\n'.format(loss_iou) + 'loss_classes: {}\n'.format(loss_classes)
				#'learning_rate: {}\n'.format(self.learning_rate.eval()) + 
				#if current_epoch != self.data.epoch:
				#	print("epoch#", current_epoch)
				#	print("Iter#", i)
				#	print(text)
				#	if self.wechat:
				#		itchat.send('epoch#{}\n'.format(current_epoch) + text, toUserName='filehelper')
					
				#	current_epoch = self.data.epoch
				#	print('{} Saving checkpoint file to: {}'.format(
				#			datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
				#			self.save_dir))
				#	self.saver.save(sess, self.ckpt_file, global_step=self.global_step)

				if (self.print_iter <= 0 ) or (i % self.print_iter == 0):
					print("epoch#", self.data.epoch)
					print("Iter#", i)
					print(text)
					if self.wechat:
						itchat.send('Iter#{}\n'.format(i) + text, toUserName='filehelper')
			
			print('{} Saving checkpoint file to: {}'.format(
						datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
						self.save_dir))
			self.saver.save(sess, self.ckpt_file, global_step=self.global_step)

			if self.wechat:
				itchat.send('DONE TRAINING', toUserName='filehelper')

	def save_cfg(self):
		with open(os.path.join(self.save_dir, 'config.txt'), 'w') as f:
			cfg_dict = cfg.__dict__
			for key in sorted(cfg_dict.keys()):
				if key[0].isupper():
					cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
					f.write(cfg_str)