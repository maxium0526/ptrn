# the folder path of your CheXpert and CheXphoto datasets
# e.g. your dataset is in /aaa/bbb/CheXpert-v1.0, then dataset_path='/aaa/bbb'
dataset_path = '/home/max/datasets'

ptrn_path = 'instances/best_iou.h5'

img_size = (224, 224)

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from utils import *
import numpy as np
import cv2 as cv

import os

# Use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ptrn = load_model(ptrn_path)

x_valid, y_valid = load_validation_data(dataset_path, dataset_name='CheXphoto-natural', size=img_size)
loss = ptrn.evaluate(preprocess_input(x_valid), y_valid)
y_pred = ptrn.predict(preprocess_input(x_valid))
ious = [IOU(y_t, y_p) for y_t, y_p in zip(y_valid, y_pred)]
iou = np.mean(ious)
print('PTRN evaluated on validation data: loss: {:.4f}, IOU: {:.4f}'.format(loss, iou))

x_valid, y_valid = load_validation_data(dataset_path, dataset_name='CheXphoto-film', size=img_size)
loss = ptrn.evaluate(preprocess_input(x_valid), y_valid)
y_pred = ptrn.predict(preprocess_input(x_valid))
ious = [IOU(y_t, y_p) for y_t, y_p in zip(y_valid, y_pred)]
iou = np.mean(ious)
print('PTRN evaluated on test data: loss: {:.4f}, IOU: {:.4f}'.format(loss, iou))