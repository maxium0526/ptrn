import pandas as pd
import os
import numpy as np
import cv2 as cv
from PIL import Image
from PIL import ImageFile
import json
from shapely.geometry import Polygon
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.stats
import imgaug.augmenters as iaa
import gc

# this function is to generate random matrix A in step 2 of training sample generation
def random_matrix():
	while True:
		C_x = np.random.uniform(0.2, 0.8)
		C_y = np.random.uniform(0.2, 0.8)
		if abs(C_x-C_y)<=0.2:
			break

	S_x = np.random.uniform(-0.1, 0.1)
	S_y = np.random.uniform(-0.1, 0.1)
	alpha = np.random.uniform(-np.pi, np.pi)
	T_x = np.random.normal(scale=0.25)
	T_y = np.random.normal(scale=0.25)
	F_x = np.random.normal(scale=0.1)
	F_y = np.random.normal(scale=0.1)

	T = [
			[1, 0, T_x],
			[0, 1, T_y],
			[0, 0, 1]
	]
	T = np.array(T, dtype='float32')

	F = [
			[0, 0, 0],
			[0, 0, 0],
			[F_x, F_y, 0]
	]
	F = np.array(F, dtype='float32')

	C = [
			[C_x, 0, 0],
			[0, C_y, 0],
			[0, 0, 1]
	]
	C = np.array(C, dtype='float32')

	S = [
			[1, S_x, 0],
			[S_y, 1, 0],
			[0, 0, 1]
	]
	S = np.array(S, dtype='float32')

	R = [
			[np.math.cos(alpha), np.math.sin(alpha), 0],
			[-np.math.sin(alpha), np.math.cos(alpha), 0],
			[0, 0, 1]
	]
	R = np.array(R, dtype='float32')

	M = [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
	]
	M = np.array(M, dtype='float32')

	M = np.matmul(C, M)
	M = np.matmul(S, M)
	M = np.matmul(R, M)
	M = np.matmul(T, M)
	M = F+M

	return M

# for test only
def certain_matrix():
	while True:
		C_x = 0.6
		C_y = 0.75
		if abs(C_x-C_y)<=0.2:
			break

	S_x = 0.07
	S_y = -0.02
	alpha = 0.31415926
	T_x = 0.1
	T_y = 0.5
	F_x = 0.03
	F_y = -0.02

	T = [
			[1, 0, T_x],
			[0, 1, T_y],
			[0, 0, 1]
	]
	T = np.array(T, dtype='float32')

	F = [
			[1, 0, 0],
			[0, 1, 0],
			[F_x, F_y, 1]
	]
	F = np.array(F, dtype='float32')

	C = [
			[C_x, 0, 0],
			[0, C_y, 0],
			[0, 0, 1]
	]
	C = np.array(C, dtype='float32')

	S = [
			[1, S_x, 0],
			[S_y, 1, 0],
			[0, 0, 1]
	]
	S = np.array(S, dtype='float32')

	R = [
			[np.math.cos(alpha), np.math.sin(alpha), 0],
			[-np.math.sin(alpha), np.math.cos(alpha), 0],
			[0, 0, 1]
	]
	R = np.array(R, dtype='float32')

	M = [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
	]
	M = np.array(M, dtype='float32')

	M = np.matmul(C, M)
	M = np.matmul(S, M)
	M = np.matmul(R, M)
	M = np.matmul(F, M)
	M = np.matmul(T, M)

	return M

# a function that converts a matrix to the 4 vertices
def matrix_to_pts(matrix, size=(224, 224)):
	M = matrix
	orig_pts = [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
	orig_pts = np.array(orig_pts, dtype='float32')
	t = np.zeros((4, 3), dtype='float32')
	for i in range(4):
		j = orig_pts[i].reshape(-1, 1)
		t[i] = np.matmul(M, j).reshape(3)
		t[i] = t[i] / t[i, 2]
	t = t[:, :-1]
	
	t = t + 1
	t = t / 2
	t = t * np.array(size, dtype='float32')

	return t

# the inverse function of above
def pts_to_matrix(pts, size=(224, 224)):
	t = pts
	t = t / np.array(size, dtype='float32')
	t = t * 2
	t = t - 1

	orig_pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
	orig_pts = np.array(orig_pts, dtype='float32')

	M = cv.getPerspectiveTransform(orig_pts, t)

	return M

# the vertices is normalized in [-1, 1]
def perspective_transform(img, matrix, size=(224, 224), fillcolor=None):
	src_pts = np.array([[0, 0],[size[0], 0],[size[0], size[1]],[0, size[1]]], dtype='float32')
	dst_pts = matrix_to_pts(matrix, size=size)
	M = cv.getPerspectiveTransform(dst_pts, src_pts)
	X_SC = img.transform(size, Image.PERSPECTIVE,M.reshape(-1)[:-1], Image.BICUBIC, fillcolor=fillcolor)
	return X_SC

def random_crop(img, size=(224, 224)):
	w, h = img.size
	if w<size[0] or h<size[1]:
		return img.resize(size)
		
	if w == size[0]:
		offsetX = 0
	else:
		offsetX = np.random.randint(0, w-size[0])

	if h == size [1]:
		offsetY = 0
	else:
		offsetY = np.random.randint(0, h-size[1])

	return img.crop((offsetX, offsetY, offsetX + size[0], offsetY + size[1]))

def composite(fg, bg, size=(224, 224)):
	new = bg.copy()
	new = random_crop(new, size=size)
	new.paste(fg, (0, 0), fg)

	return np.array(new)

aug = iaa.Sequential([
	iaa.Add((-40, 40)),
	iaa.Multiply((0.7, 1.3)),		
	iaa.Sometimes(0.1, iaa.AverageBlur(k=2)),
	iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-40, 40), per_channel=True)),
	iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)),
	iaa.Sometimes(0.5, iaa.pillike.EnhanceColor()),
	iaa.Sometimes(0.5, iaa.pillike.EnhanceContrast()),
	iaa.Sometimes(0.5, iaa.pillike.EnhanceBrightness()),
	iaa.Sometimes(0.5, iaa.pillike.EnhanceSharpness()),
	iaa.Sometimes(0.5, iaa.JpegCompression(compression=(70, 99))),
	iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)))
	], random_order=True)

def augment(img):
	return aug(images=img)

# generate a batch of training sample
def generate_training_sample_batch(dataset_path, df_cxrs, bg_paths, batch_size=32, size=(224, 224)):

	# the series of CXR images
	df_selected_cxrs = df_cxrs.sample(n=batch_size)

	# load the random image
	bg_path = bg_paths.sample(n=1).iloc[0]

	try:
		bg = Image.open(bg_path)
	except:
		bg = Image.new(mode="RGBA", size=size)
		print('Cannot load the bg.')

	I_FBs = []
	outs =[]
	for i, row in df_selected_cxrs.iterrows():
		# load a CXR image
		img_path = os.path.join(dataset_path, row['Path'])
		X = Image.open(img_path).convert('RGBA')
		X = X.resize(size)

		# step 1: screen synthesis
		if np.random.rand()<0.3:
			screen_synthesis = True
		else:
			screen_synthesis = False

		if screen_synthesis is True:
			t = np.random.rand() * 0.6
			b = np.random.rand() * 0.6
			l = np.random.rand() * 0.6
			r = np.random.rand() * 0.6

			A_SC = [
				[1-(l+r)/2, 0, (l-r)/2],
				[0, 1-(t+b)/2, (t-b)/2],
				[0, 0, 1],
			]
			A_SC = np.array(A_SC, dtype='float32')

			color_r = np.random.randint(0, 20)
			color_g = np.random.randint(0, 20)
			color_b = np.random.randint(0, 20)
	
			X_SC = perspective_transform(X, A_SC, size=size, fillcolor=(color_r, color_g, color_b, 255))

		# step 2: perspective transformation
		A = random_matrix()

		if screen_synthesis is True:
			I_F = perspective_transform(X_SC, np.matmul(A, np.linalg.inv(A_SC)), size=size, fillcolor=(0, 0, 0, 0))
		else:
			I_F = perspective_transform(X, A, size=size, fillcolor=(0, 0, 0, 0))	

		# step 3: adding background
		I_FB = composite(I_F, bg, size=size)

		I_FB = np.array(I_FB)
		I_FB = cv.cvtColor(I_FB, cv.COLOR_RGBA2RGB)
		# ------
		I_FBs.append(I_FB)
		outs.append(A.reshape(-1)[:-1])

		# del X, I_FB, I_F
		# if screen_synthesis is True:
		# 	del X_SC
		# gc.collect()

# step 4: Decreasing quality
	I_FBs = np.array(I_FBs).reshape(-1, *size, 3)
	if aug is not None:
		I_photos = augment(I_FBs)
	else:
		I_photos = I_FBs
	outs = np.array(outs).reshape(-1, 8)

	# del bg

	return I_photos, outs

def load_validation_data(dataset_path, dataset_name='CheXphoto-natural', size=(224, 224)):
	df_valid = pd.read_csv(os.path.join(dataset_path, dataset_name, 'valid.csv'))['Path']
	
	imgs = []
	outs =[]
	for i, path in df_valid.iteritems():
		img_path = os.path.join(dataset_path, path)
		label_path = '.'.join(img_path.split('.')[:-1]) + '.json'
		img = cv.imread(img_path, -1)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		h, w, c = img.shape

		f = open(label_path)
		label = json.load(f)
		label = label['shapes'][0]['points']
		f.close()
					
		pts = np.array([
			[label[3][0] * (size[0]/w), label[3][1] * (size[1]/h)],
			[label[0][0] * (size[0]/w), label[0][1] * (size[1]/h)],
			[label[1][0] * (size[0]/w), label[1][1] * (size[1]/h)],
			[label[2][0] * (size[0]/w), label[2][1] * (size[1]/h)],				
			], dtype='float32')

		img = cv.resize(img, size, interpolation=cv.INTER_AREA)	

		A = pts_to_matrix(pts, size=size)
		imgs.append(img)
		outs.append(A.reshape(-1)[:-1])

	imgs = np.array(imgs).reshape(-1, *size, 3)
	outs = np.array(outs).reshape(-1, 8)

	return imgs, outs

def apply(img, matrix, size=None):
	if size is None:
		h, w = img.shape[0], img.shape[1]
	else:
		h, w = size
	pts = matrix_to_pts(matrix, size=(w, h))

	img = perspective_transform(Image.fromarray(img), np.linalg.inv(matrix), fillcolor=0)

	return np.array(img)

def draw_pts(img, matrix):
	pts = matrix_to_pts(matrix)

	new = np.array(img)
	new = cv.drawContours(new, [pts.astype('int')], -1, color=(0, 255, 255), thickness=cv.FILLED)

	for pt in pts:
		new = cv.circle(new, (int(pt[0]), int(pt[1])), 9, (255, 150, 150), -1)
	return new

def IOU(y_true, y_pred):
	m_true = np.concatenate((np.array(y_true), np.ones((1)))).reshape(3, 3)
	m_pred = np.concatenate((np.array(y_pred), np.ones((1)))).reshape(3, 3)

	pts_true = matrix_to_pts(m_true, size=(224, 224))
	pts_pred = matrix_to_pts(m_pred, size=(224, 224))

	polygon1_shape = Polygon(pts_true)
	polygon2_shape = Polygon(pts_pred)

	# Calculate intersection and union, and the IOU
	polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
	polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
	return polygon_intersection / polygon_union

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h
