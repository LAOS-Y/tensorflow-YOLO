import tensorflow as tf
import tensorlayer as tl

import cv2
import numpy as np
import utils
import model
import config as cfg
import train

net = model.Net()

data = utils.pascal_voc("train", rebuild = False)

print("Begin")

solver = train.Solver(net, data, wechat = True)

solver.train()