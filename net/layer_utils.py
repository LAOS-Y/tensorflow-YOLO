import tensorflow as tf

def coor(box):
	# x,y,w,h
	with tf.name_scope("Coordinates"):
		left = box[:, :, :, :, 0] - box[:, :, :, :, 2] / 2.0
		right = box[:, :, :, :, 0] + box[:, :, :, :, 2] / 2.0
		bottom = box[:, :, :, :, 1] - box[:, :, :, :, 3] / 2.0
		top = box[:, :, :, :, 1] + box[:, :, :, :, 3] / 2.0
		return left, right, bottom, top

def iou(box1, box2):
	with tf.name_scope("IOU"):
		left1, right1, bottom1, top1 = coor(box1)
		left2, right2, bottom2, top2 = coor(box2)
		inter_left = tf.maximum(left1, left2)
		inter_bottom = tf.maximum(bottom1, bottom2)
		inter_right = tf.minimum(right1, right2)
		inter_top = tf.minimum(top1, top2)
		intersection = (inter_right - inter_left) * (inter_top - inter_bottom)
		union = box2[:, :, :, :, 2] * box2[:, :, :, :, 3] + box1[:, :, :, :, 2] * box1[:, :, :, :, 3] - intersection
		return intersection / union

def select_box_cell(box1, box2, ground_truth):
	with tf.name_scope("select_box_cell"):
		iou1 = iou(box1[1:], ground_truth)
		iou2 = iou(box2[1:], ground_truth)
		return tf.cond(iou1[i] > iou2[i], true_fn = lambda: box1, false_fn = lambda: box2)

def select_box_image(box_image, ground_truth):
	with tf.name_scope("select_box_image"):
		H, W = tf.shape(box_image)[0], tf.shape(box_image)[1]
		
