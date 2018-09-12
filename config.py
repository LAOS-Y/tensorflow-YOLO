import os

LEARNING_RATE = 0.0001
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True
BATCH_SIZE = 5
MAX_ITER = 15000
SUMMARY_ITER = 100
PRINT_ITER = 150
SAVE_ITER = 10

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

#OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
OUTPUT_DIR = '/output'

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
			'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
			'train', 'tvmonitor']

IMAGE_SIZE = 448

CELL_SIZE = 7

FLIPPED = True