import numpy
import tensorflow as tf

def coor(box):
	# x,y,w,h
	with tf.name_scope("Coordinates"):
		left = box[0] - box[2] / tf.constant(2.0)
		right = box[0] + box[2] / tf.constant(2.0)
		bottom = box[1] - box[3] / tf.constant(2.0)
		top = box[1] + box[3] / tf.constant(2.0)
		return left, right, bottom, top

def iou(bounding_box, ground_truth):
	with tf.name_scope("IOU"):
		box_left, box_right, box_bottom, box_top = coor(bounding_box)
		truth_left, truth_right, truth_bottom, truth_top = coor(ground_truth)
		left = tf.maximum(box_left, truth_left)
		bottom = tf.maximum(box_bottom, truth_bottom)
		right = tf.minimum(box_right, truth_right)
		top = tf.minimum(box_top, truth_top)
		intersection = (right - left) * (top - bottom)
		#union = (box_right - box_left) * (box_top - box_bottom) + (truth_right - truth_left) * (truth_top - truth_bottom) - intersection
		union = ground_truth[:, 2] * ground_truth[:, 3] + bounding_box[:, 2] * bounding_box[:, 3] - intersection
		return intersection / union

def select_box_cell(box1, box2, ground_truth):
	with tf.name_scope("select_box_cell"):
		iou1 = iou(box1[1:], ground_truth)
		iou2 = iou(box2[1:], ground_truth)
		return tf.cond(iou1[i] > iou2[i], true_fn = lambda: box1, false_fn = lambda: box2)

def select_box_image(box_image, ground_truth):
	with tf.name_scope("select_box_image"):
		H, W = tf.shape(box_image)[0], tf.shape(box_image)[1]
		
