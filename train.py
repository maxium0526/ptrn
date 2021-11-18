# the folder path of your CheXpert and CheXphoto datasets
# e.g. your dataset is in /aaa/bbb/CheXpert-v1.0, then dataset_path='/aaa/bbb'
dataset_path = '/home/max/datasets'

# where is your folder that contains random images
ms_coco_path = '/home/max/datasets/MSCOCO/test2017'

img_size = (224, 224)

batch_size = 32

max_epochs = 10000

steps_per_epoch = 100

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import os
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf
import glob
from utils import *
import gc

df_cxrs = pd.read_csv(os.path.join(dataset_path, 'CheXpert-v1.0-small', 'train.csv'))

bg_paths = glob.glob(os.path.join(ms_coco_path, '*.jpg'), recursive=True)
bg_paths = pd.Series(bg_paths)

def STRN():
	model_in = Input(shape=(*img_size, 3))
	backbone = DenseNet201(include_top=False, weights='imagenet', pooling='avg')
	x = backbone(model_in)
	model_out = Dense(8, activation='linear')(x)

	model = Model(inputs=[model_in], outputs=[model_out])
	model.compile(
		optimizer=Adam(learning_rate=1e-5),
		loss='mse',
		)
	return model

strn = STRN()

x_valid, y_valid = load_validation_data(dataset_path, dataset_name='CheXphoto-natural', size=img_size)

train_record = pd.DataFrame(columns=['train_loss', 'train_iou', 'valid_loss', 'valid_iou'])

if not os.path.exists('rectified_photograph_samples'):
	os.makedirs('rectified_photograph_samples')

# record the best score
best_valid_loss = 999
best_valid_iou = 0
for i in range(1, max_epochs+1):
	record = {}

	train_losses = []
	train_ious = []
	for j in range(1,steps_per_epoch+1):
		# train and calculate train_loss
		x_train, y_train = generate_training_sample_batch(dataset_path, df_cxrs, bg_paths, batch_size=batch_size, size=img_size)
		train_loss = strn.train_on_batch(x=tf.convert_to_tensor(preprocess_input(x_train)), y=tf.convert_to_tensor(y_train))
		train_losses.append(train_loss)

		# calculate train_iou
		ious = []
		y_pred = strn.predict_on_batch(tf.convert_to_tensor(preprocess_input(x_train)))
		for k in range(y_train.shape[0]):
			try:
				iou = IOU(y_train[k], y_pred[k])
			except:
				print('Cannot calculate IoU')
				iou = 0
			ious.append(iou)
		train_ious.append(np.mean(ious))
		print('Step: {:d}/{:d} loss: {:.4f}'.format(j, 100, train_loss), end='\r')

	train_loss = np.mean(train_losses)
	train_iou = np.mean(train_ious)

	record['train_loss'] = train_loss
	record['train_iou'] = train_iou

	# calculate valid_loss
	valid_loss = strn.evaluate(x=tf.convert_to_tensor(preprocess_input(x_valid)), y=tf.convert_to_tensor(y_valid), verbose=0)
	record['valid_loss'] = valid_loss

	# calculate valid_iou
	valid_ious = []
	for x, y in zip(x_valid, y_valid):
		y_pred = strn.predict_on_batch(tf.convert_to_tensor(preprocess_input(np.expand_dims(x, axis=0))))[0]
		try:
			iou = IOU(y, y_pred)		
		except:
			print('Cannot calculate IoU')
			iou = 0
		valid_ious.append(iou)
	valid_iou = np.mean(valid_ious)
	record['valid_iou'] = valid_iou

	print('Epoch: {:d} Step: {:d}/{:d} train_loss: {:.4f}, train_iou: {:.4f}, valid_loss: {:.4f}, valid_iou: {:.4f}'.format(i, j, steps_per_epoch, train_loss, train_iou, valid_loss, valid_iou))

	train_record = train_record.append(record, ignore_index=True)

	# plot a rectified photographs to verify the training
	y = strn.predict_on_batch(tf.convert_to_tensor(preprocess_input(np.expand_dims(x_valid[0], axis=0))))[0]
	try:
		# cv.imwrite('plot/{:d}.jpg'.format(i), draw_pts(x_valid[0], y))
		cv.imwrite('rectified_photograph_samples/{:d}.jpg'.format(i), apply(x_valid[0], out_to_matrix(y)))
	except:
		print('Cannot save the image.')

	# plot a training sample.
	try:
		cv.imwrite('sample.jpg', x_train[0])
	except:
		print('Cannot save the image.')

	# save training log
	try:
		train_record.to_csv('log.csv')
	except:
		print('Cannot save the log.')
		
	if valid_loss <= best_valid_loss:
		best_valid_loss	= valid_loss
		try:
			strn.save('instances/best_loss.h5')
		except:
			print('Cannot save the model.')

	if valid_iou >= best_valid_iou:
		best_valid_iou	= valid_iou
		try:
			strn.save('instances/best_iou.h5')
		except:
			print('Cannot save the model.')

	clear_session()
	gc.collect()