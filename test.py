import tensorflow as tf
import tensorlayer as tl

import cv2
import numpy as np
import utils
import model
import config as cfg

net = model.Net()

X, y = np.random.randn(1, 448, 448, 3), np.random.randn(1,7, 7, 25)
feed_dict = {net.X: X, net.y_hat: y}

print("Begin")

with tf.Session() as sess:
	initial_learning_rate = cfg.LEARNING_RATE
	decay_rate = cfg.DECAY_RATE
	decay_steps = cfg.DECAY_STEPS
	staircase = cfg.STAIRCASE
	global_step = tf.train.create_global_step()
	learning_rate = tf.train.exponential_decay(initial_learning_rate,
		global_step,
		decay_steps,
		decay_rate,
		staircase,
		name='learning_rate')
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(net.loss, global_step = global_step)
	sess.run(tf.global_variables_initializer())
	print("INIT DONE")

	loss_coor, loss_confidence, loss_classes = sess.run([net.loss_coor, net.loss_confidence, net.loss_classes], feed_dict = feed_dict)
	print("loss_coor: ", loss_coor)
	print("loss_confidence: ", loss_confidence)
	print("loss_classes: ", loss_classes)