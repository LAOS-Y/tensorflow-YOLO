import numpy as np
import matplotlib as plot
import config as cfg

def decodeLabel(coor, classes):
	classes_list = cfg.CLASSES
	image_size = cfg.IMAGE_SIZE
	cell_size = cfg.CEll_SIZE
	cell_width = image_size / cell_size
	offset_x = np.array([range(cell_size) for i in range(cell_size)])
	offset_y = offset_x.T
	offset_x = off_set_x.reshape(1, cell_size, cell_size)
	offset_y = off_set_y.reshape(1, cell_size, cell_size)
	x, y = coor[:, :, :, 0], coor[:,:]
