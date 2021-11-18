ptrn_path = 'instances/best_iou.h5'

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from utils import *
import numpy as np
import cv2 as cv
import sys
import os

# disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

img_name = sys.argv[1]

ptrn = load_model(ptrn_path)

img = cv.imread(img_name)
model_in = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
model_in = cv.cvtColor(model_in, cv.COLOR_BGR2RGB)
model_in = np.expand_dims(model_in, axis=0)
model_in = preprocess_input(model_in)
out = ptrn.predict_on_batch(model_in)[0]

rectified_img = apply(img, out_to_matrix(out))
cv.imwrite('rectified_'+img_name, rectified_img)

region = np.ones((224, 224, 3), dtype='float32')
region = draw_pts(region, out_to_matrix(out))
cv.imwrite('region_'+img_name, region)