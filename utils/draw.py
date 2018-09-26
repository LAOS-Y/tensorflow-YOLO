import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config as cfg
import utils.math as math
import colorsys
import random
import cv2

def decodeCoor(coor):
	image_size = cfg.IMAGE_SIZE
	cell_size = cfg.CELL_SIZE
	cell_width = image_size / cell_size
	offset_x = np.array([range(cell_size) for i in range(cell_size)])
	offset_y = offset_x.T
	x, y = coor[:, :, 0], coor[:, :, 1]
	w, h = coor[:, :, 2], coor[:, :, 3]
	
	x = (x + offset_x) * cell_width
	y = (y + offset_y) * cell_width
	w = w * cell_width
	h = h * cell_width

	return x, y, w, h


def generateRectList(label, tag_num, min_c, nms_k, isTruth):
	tag = cfg.CLASSES
	cell_size = cfg.CELL_SIZE

	hsv_tuples = [(x/float(len(tag)), 1., 1.)  for x in range(len(tag))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
	random.seed(10101)
	random.shuffle(colors)

	if not isTruth:
		confidence = np.reshape(label[:, :, :2], [cell_size, cell_size, 2, 1])
		box = np.reshape(label[:, :, 2:10], [cell_size, cell_size, 2, 4])
		classes = label[:, :, 10:]
		classes_prob = math.softmax(classes, axis = 2)

		mask = np.max(confidence, axis = 2, keepdims = True) <= confidence
		mask = np.reshape(mask, (7, 7, 2, 1))
		confidence = np.reshape(confidence[mask], (cell_size, cell_size))
		box = np.reshape(box[np.tile(mask, (1, 1, 1, 4))], (cell_size, cell_size, 4))
	else:
		confidence = label[:, :, 0]
		box = label[:, :, 1:5]
		classes_prob = label[:, :, 5:]
	x, y, w, h = decodeCoor(box)

	rect_list = []
	for idx_y in range(cell_size):
		for idx_x in range(cell_size):
			if confidence[idx_y, idx_x] < min_c:
				continue
			prob_vec = classes_prob[idx_y, idx_x, :]
			sorted_classses_idx = np.argsort(prob_vec)
			
			idx = sorted_classses_idx[-1]
			text = " {}: {}".format(tag[idx], prob_vec[idx])

			box_coor = (x[idx_y, idx_x],
						y[idx_y, idx_x],
						w[idx_y, idx_x],
						h[idx_y, idx_x])
			rect_list.append((colors[idx], confidence[idx_y, idx_x], box_coor, text))

	return nms(rect_list, nms_k)

def mergeRect(image, label, tag_num = 3, min_c = 0.45, nms_k = 0.8, isTruth = True):
	_, ax = plt.subplots(1)
	image = (image + 1) / 2
	ax.imshow(image)
	rect_list = generateRectList(label, tag_num, min_c, nms_k, isTruth)
	for rect in rect_list:
		box = rect[2]
		x1, y1 = box[0] - box[2] / 2, box[1] - box[3] / 2
		x2, y2 = box[0] + box[2] / 2, box[1] + box[3] / 2

		thick = int((box[2] + box[3]) / 300)

		if box[1] < 20:
			text_loc = (int(x1 + 2), int(y1 + 15))
		else:
			text_loc = (int(x1), int(y1 - 10))

		cv2.rectangle(image, 
			(int(x1), int(y1)),
			(int(x2), int(y2)),
			rect[0],
			thick)
		cv2.putText(image, rect[3], text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * box[3], (255,255,255), thick//3)
		#cv2.imshow('fuck', image)
		return image

def nms(rect_list, nms_k):
	rect_list = sorted(rect_list, key = lambda x:x[1], reverse = True)
	selected = []
	while (len(rect_list)):
		selected.append(rect_list.pop(0))
		for i in range(len(rect_list) - 1, -1, -1):
			box1, box2 = selected[-1][2], rect_list[i][2]	
			if math.iou2(box1, box2) > nms_k:
				rect_list.pop(i)
	return selected