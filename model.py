import numpy as np
import tensorflow as tf
import tensorlayer as tl
from net import *


class Net():
	def __init__(self):
		self.X = tf.placeholder(dtype = tf.float32, shape = [None, 448, 448, 3], name = "X")
		self.y_hat = tf.placeholder(dtype = tf.float32, shape = [None, 7, 7, 25], name = 'y_hat')

		with tf.variable_scope("Input"):
			input_layer = tl.layers.InputLayer(self.X)

		with tf.variable_scope("Layer1"):
			conv_layer = conv(input_layer, 64, (7, 7), (2, 2), "conv1")
			pool_layer = maxpool(conv_layer, "pool1")

		with tf.variable_scope("Layer2"):
			conv_layer = conv(pool_layer, 192, (3, 3), (1, 1), "conv2")
			pool_layer = maxpool(conv_layer, "pool2")

		with tf.variable_scope("Layer3"):
			conv_layer = conv(pool_layer, 128, (1, 1), (1, 1), "conv3")
			conv_layer = conv(conv_layer, 256, (3, 3), (1, 1), "conv4")
			conv_layer = conv(conv_layer, 256, (1, 1), (1, 1), "conv5")
			conv_layer = conv(conv_layer, 512, (3, 3), (1, 1), "conv6")
			pool_layer = maxpool(conv_layer, "pool3")

		with tf.variable_scope("Layer4"):
			conv_layer = pool_layer
			for i in range(4):
				conv_layer = conv(conv_layer, 256, (1, 1), (1, 1), "conv%d"%(7 + i * 2))
				conv_layer = conv(conv_layer, 512, (3, 3), (1, 1), "conv%d"%(8 + i * 2))
			conv_layer = conv(conv_layer, 512, (1, 1), (1, 1), "conv15")
			conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv16")
			pool_layer = maxpool(conv_layer, "pool4")

		with tf.variable_scope("Layer5"):
			conv_layer = pool_layer
			for i in range(2):
				conv_layer = conv(conv_layer, 512, (1, 1), (1, 1), "conv%d"%(17 + i * 2))
				conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv%d"%(18 + i * 2))

			conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv21")
			conv_layer = conv(conv_layer, 1024, (3, 3), (2, 2), "conv22")

		with tf.variable_scope("Layer6"):
			conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv23")
			conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv24")

		with tf.variable_scope("Layer7"):
			dense_layer = dense(flatten(conv_layer, "flatten"), 4096, "dense1")

		with tf.variable_scope("Output"):
			dense_layer = dense(dense_layer, 7 * 7 *30, "dense2")
			self.network = reshape(dense_layer, (-1, 7, 7, 30), "reshape1")
			self.y = self.network.outputs
			print(type(self.y))

		with tf.variable_scope("Confidence"):
			self.confidence = tf.sigmoid(self.y[:, :, :, :2])

		with tf.variable_scope("Box"):
			self.box = self.y[:, :, :, 2:10]
			self.box = tf.reshape(self.box, [-1, 7, 7, 2, 4])

			box_xy = tf.sigmoid(self.box[:, :, :, :, :2])
			box_wh = tf.exp(self.box[:, :, :, :, 2:])
			self.box = tf.concat([box_xy, box_wh], axis = 4)
		
		with tf.variable_scope("Classes"):
			self.classes = self.y[:, :, :, 10:]

		with tf.variable_scope("Confidence_hat"):
			confidence_hat = self.y_hat[:, :, :, :1]
			confidence_hat = tf.tile(confidence_hat, [1, 1, 1, 2])

		with tf.variable_scope("Box_hat"):
			box_hat = self.y_hat[:, :, :, 1:5]
			box_hat = tf.reshape(box_hat, [-1, 7, 7, 1, 4])
			box_hat = tf.tile(box_hat, [1, 1, 1, 2, 1])
		
		with tf.variable_scope("Classes_hat"):
			classes_hat = self.y_hat[:, :, :, 5:]

		with tf.variable_scope("Loss"):
			box_iou = iou(self.box, box_hat)
			mask_obj = tf.reduce_max(box_iou, axis = 3, keepdims = True) <= box_iou
			mask_obj = tf.cast(mask_obj, tf.float32)
			mask_noobj = tf.ones_like(mask_obj) - mask_obj
			
			lambda_coord = 5
			lambda_noobj = 0.5

			with tf.variable_scope("Loss_coor"):
				with tf.variable_scope("Loss_coor"):
					self.loss_coor = lambda_coord * tf.reduce_mean(tf.reduce_sum(mask_obj * (tf.squared_difference(self.box[:, :, :, :, 0], box_hat[:, :, :, :, 0], name = "diff_x")
																							+ tf.squared_difference(self.box[:, :, :, :, 1], box_hat[:, :, :, :, 1], name = "diff_y")),
																				axis = [1, 2, 3]))
					self.loss_coor += lambda_coord * tf.reduce_mean(tf.reduce_sum(mask_obj * (tf.squared_difference(tf.sqrt(self.box[:, :, :, :, 2]), tf.sqrt(box_hat[:, :, :, :, 2]), name = "diff_w")
																							+ tf.squared_difference(tf.sqrt(self.box[:, :, :, :, 3]), tf.sqrt(box_hat[:, :, :, :, 3]), name = "diff_h")),
																				axis = [1, 2, 3]))

			with tf.variable_scope("Loss_iou"):
				self.loss_iou = tf.reduce_mean(tf.reduce_sum(mask_obj * tf.squared_difference(self.confidence, confidence_hat), axis = [1, 2, 3]))
				self.loss_iou += lambda_noobj * tf.reduce_mean(tf.reduce_sum(mask_noobj * tf.squared_difference(self.confidence, confidence_hat), axis = [1, 2, 3]))
			
			with tf.variable_scope("Loss_classes"):
				self.loss_classes = tf.reduce_mean(tf.reduce_sum(confidence_hat[:, :, :, :1] * tf.squared_difference(self.classes, classes_hat), axis = [1, 2, 3]))

			self.loss = self.loss_coor + self.loss_iou + self.loss_classes

			tf.summary.scalar('loss_coor', self.loss_coor)
			tf.summary.scalar('loss_iou', self.loss_iou)
			tf.summary.scalar('loss_classes', self.loss_classes)
			tf.summary.scalar('loss_total', self.loss)

#net = Net()