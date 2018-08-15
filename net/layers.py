import tensorflow as tf
import tensorlayer as tl
import net.af as af

conv = lambda in_layer, filter_num, filter_shape, _strides, _name: tl.layers.Conv2d(
	in_layer,
	n_filter = filter_num,
	act = af.leaky_relu,
	filter_size = filter_shape,
	padding = "SAME",
	strides = _strides,
	name = _name)

dense = lambda in_layer, h, _name: tl.layers.DenseLayer(
	in_layer,
	n_units = h,
	act = af.leaky_relu,
	name = _name)

flatten = lambda in_layer, _name: tl.layers.FlattenLayer(
	in_layer,
	name = _name)

maxpool = lambda in_layer, _name: tl.layers.MaxPool2d(
	in_layer,
	filter_size = (2, 2),
	padding = "SAME",
	strides = (2, 2),
	name = _name)

reshape = lambda in_layer, shape, _name: tl.layers.ReshapeLayer(
	in_layer,
	shape,
	name = _name)