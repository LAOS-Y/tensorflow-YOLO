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
			conv_bn_layer = conv_bn(input_layer, 64, (7, 7), (2, 2), "conv_bn1")
			pool_layer = maxpool(conv_bn_layer, "pool1")

		with tf.variable_scope("Layer2"):
			conv_bn_layer = conv_bn(pool_layer, 192, (3, 3), (1, 1), "conv_bn2")
			pool_layer = maxpool(conv_bn_layer, "pool2")

		with tf.variable_scope("Layer3"):
			conv_bn_layer = conv_bn(pool_layer, 128, (1, 1), (1, 1), "conv_bn3")
			conv_bn_layer = conv_bn(conv_bn_layer, 256, (3, 3), (1, 1), "conv_bn4")
			conv_bn_layer = conv_bn(conv_bn_layer, 256, (1, 1), (1, 1), "conv_bn5")
			conv_bn_layer = conv_bn(conv_bn_layer, 512, (3, 3), (1, 1), "conv_bn6")
			pool_layer = maxpool(conv_bn_layer, "pool3")

		with tf.variable_scope("Layer4"):
			conv_bn_layer = pool_layer
			for i in range(4):
				conv_bn_layer = conv_bn(conv_bn_layer, 256, (1, 1), (1, 1), "conv_bn%d"%(7 + i * 2))
				conv_bn_layer = conv_bn(conv_bn_layer, 512, (3, 3), (1, 1), "conv_bn%d"%(8 + i * 2))
			conv_bn_layer = conv_bn(conv_bn_layer, 512, (1, 1), (1, 1), "conv_bn15")
			conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (1, 1), "conv_bn16")
			pool_layer = maxpool(conv_bn_layer, "pool4")

		with tf.variable_scope("Layer5"):
			conv_bn_layer = pool_layer
			for i in range(2):
				conv_bn_layer = conv_bn(conv_bn_layer, 512, (1, 1), (1, 1), "conv_bn%d"%(17 + i * 2))
				conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (1, 1), "conv_bn%d"%(18 + i * 2))

			conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (1, 1), "conv_bn21")
			conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (2, 2), "conv_bn22")

		with tf.variable_scope("Layer6"):
			conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (1, 1), "conv_bn23")
			conv_bn_layer = conv_bn(conv_bn_layer, 1024, (3, 3), (1, 1), "conv_bn24")

		with tf.variable_scope("Layer7"):
			transpose_layer = transpose(conv_bn_layer, [0, 3, 1, 2], "transpose")
			dense_layer = dense(flatten(transpose_layer, "flatten"), 4096, "dense1")

		with tf.variable_scope("Output"):
			self.network = dense(dense_layer, 7 * 7 *30, "dense2")
			self.y = self.network.outputs

		with tf.variable_scope("Confidence"):
			self.confidence = tf.sigmoid(self.y[:, :(7 * 7 * 2)])
			self.confidence = tf.reshape(self.confidence, [-1, 7, 7, 2])
			
		with tf.variable_scope("Box"):
			self.box = self.y[:, (7 * 7 * 2):(7 * 7 * 2 * 4 + 7 * 7 * 2)]
			self.box = tf.reshape(self.box, [-1, 7, 7, 2, 4])

			box_xy = tf.sigmoid(self.box[:, :, :, :, :2])
			box_wh = tf.exp(self.box[:, :, :, :, 2:])
			self.box = tf.concat([box_xy, box_wh], axis = 4)
		
		with tf.variable_scope("Classes"):
			self.classes = self.y[:, (7 * 7 * 2 * 4 + 7 * 7 * 2):]
			self.classes = tf.reshape(self.classes, [-1, 7, 7, 20])

		with tf.variable_scope("Reshaped_output"):
			self.y = tf.concat([self.confidence,
								tf.reshape(self.box, [-1, 7, 7, 2 * 4]),
								self.classes], axis = -1)

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
			mask_obj = tf.cast(mask_obj, tf.float32) * confidence_hat
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
				self.loss_classes = tf.reduce_mean(tf.reduce_sum(confidence_hat[:, :, :, 0] * tf.nn.softmax_cross_entropy_with_logits(logits = self.classes, labels = classes_hat), axis = [1, 2]))
			self.loss = self.loss_coor + self.loss_iou + self.loss_classes

			tf.summary.scalar('loss_coor', self.loss_coor)
			tf.summary.scalar('loss_iou', self.loss_iou)
			tf.summary.scalar('loss_classes', self.loss_classes)
			tf.summary.scalar('loss_total', self.loss)
