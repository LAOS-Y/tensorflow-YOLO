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


def coor2(box):
	# x,y,w,h
	left = box[0] - box[2] / 2.0
	right = box[0] + box[2] / 2.0
	bottom = box[1] - box[3] / 2.0
	top = box[1] + box[3] / 2.0
	return left, right, bottom, top

def iou2(box1, box2):
	left1, right1, bottom1, top1 = coor2(box1)
	left2, right2, bottom2, top2 = coor2(box2)
	inter_left = max(left1, left2)
	inter_bottom = max(bottom1, bottom2)
	inter_right = min(right1, right2)
	inter_top = min(top1, top2)
	intersection = (inter_right - inter_left) * (inter_top - inter_bottom)
	union = box2[2] * box2[3] + box1[2] * box1[3] - intersection
	return intersection / union