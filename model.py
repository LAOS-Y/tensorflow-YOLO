import numpy as np
import tensorflow as tf
import tensorlayer as tl
from net import *

class Net():

	def __init__(self):
		with tf.Session() as sess:
			with tf.variable_scope("Input"):
				X_data = tf.placeholder(dtype = tf.float32, shape = [None, 448, 448, 3])
				self.X = tl.layers.InputLayer(X_data)

			with tf.variable_scope("Layer1"):
				conv_layer = conv(self.X, 64, (7, 7), (2, 2), "conv1")
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

					writer=tf.summary.FileWriter('./graph', sess.graph)
					writer.close()

				conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv21")
				conv_layer = conv(conv_layer, 1024, (3, 3), (2, 2), "conv22")

			with tf.variable_scope("Layer6"):
				conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv23")
				conv_layer = conv(conv_layer, 1024, (3, 3), (1, 1), "conv24")

			with tf.variable_scope("Layer7"):
				dense_layer = dense(flatten(conv_layer,"flatten"), 4096, "dense1")

			with tf.variable_scope("Output"):
				dense_layer = dense(dense_layer, 7 * 7 *30, "dense2")
				self.output_layer = reshape(dense_layer, (-1, 7, 7, 30), "reshape1")
				self.y = self.output_layer.outputs
				print(type(self.y))

			with tf.variable_scope("Loss"):
				lambda_coord = 5
				lambda_noobj = 0.5
				N = tf.shape(y)[0]
				for i in range(N):
					tf.cond()

			writer=tf.summary.FileWriter('./graph', sess.graph)
			writer.close()

			
model = Net()
