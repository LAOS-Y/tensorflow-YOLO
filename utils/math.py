import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def softmax(x, axis):
	x_max = np.max(x, axis = axis, keepdims = 1)
	offset_x = x - x_max
	exp_offset_x = np.exp(offset_x)
	x_sum = np.sum(exp_offset_x, axis = axis, keepdims = 1)
	return exp_offset_x / x_sum