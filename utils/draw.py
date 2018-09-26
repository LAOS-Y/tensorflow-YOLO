import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config as cfg
import utils.math as math

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
	
	x = x - w / 2
	y = y - h / 2
	return x, y, w, h


def generateRectList(label, tag_num, min_c, nms_k, isTruth):
	#return patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
	tag = cfg.CLASSES
	cell_size = cfg.CELL_SIZE
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
			#sorted_classes = classes_prob[sorted_classses_idx]			
			#tag_and_prob = [(tag[20 - i - 1], sorted_classes) for i in range(20)]
			text = ""
			for i in range(tag_num):
				idx = sorted_classses_idx[-1 * i - 1]
				text += " {}: {}".format(tag[idx], prob_vec[idx])
			#print(text)
			rect = patches.Rectangle((x[idx_y, idx_x], y[idx_y, idx_x]),
										w[idx_y, idx_x], h[idx_y, idx_x],
										linewidth = 2,
										edgecolor = 'r',
										facecolor = 'none',
										label = text)
			box_coor = (x[idx_y, idx_x] + w[idx_y, idx_x] / 2,
						y[idx_y, idx_x] + h[idx_y, idx_x] / 2,
						w[idx_y, idx_x],
						h[idx_y, idx_x])
			rect_list.append((rect, confidence[idx_y, idx_x], box_coor, text))

	return nms(rect_list + rect_list, nms_k)

def mergeRect(image, label, tag_num = 3, min_c = 0.45, nms_k = 0.8, isTruth = True):
	_, ax = plt.subplots(1)
	image = (image + 1) / 2
	ax.imshow(image)
	rect_list = generateRectList(label, tag_num, min_c, nms_k, isTruth)
	for rect in rect_list:
		print('confidence: {}'.format(rect[1]), rect[3])
		ax.add_patch(rect[0])
	plt.show()

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