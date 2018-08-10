import tensorflow as tf
import tensorlayer as tl
import numpy as np
from net import *

with tf.Session() as sess:
	"""
	x = tf.constant(
		[[[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
		[[9, 10], [11, 12]]],
		[[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
		[[9, 10], [11, 12]]]]
	)
	net = tl.layers.InputLayer(x, name='input')
	net = tl.layers.FlattenLayer(net, name='flatten')
	print(tf.shape(x).eval())
	print(net.outputs.eval())
	net = tl.layers.ReshapeLayer(net, [-1, 3, 2, 2])
	print(net.outputs.eval())
	"""
	box = tf.constant([2, 4, 2, 4], dtype = tf.float32)
	truth = tf.constant([3, 2, 2, 4], dtype = tf.float32)

	iou1 = iou(box, truth)
	print(iou1.eval())






			
