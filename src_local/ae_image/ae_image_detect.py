# -*- coding: utf-8 -*-
# 実行方法　python3 this.py datafile

import os
import sys
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.utils import load_img, img_to_array
#from PIL import Image

# 入力データサイズ
INPUT_SIZE = 64
IMAGE_CHANNEL = 3

ModelFile = "./ae_image_model.h5"

if not os.path.exists(ModelFile):
    print("Modelfile not exist.")
    exit()

args = sys.argv
Datafile = args[1]
if not os.path.exists(Datafile):
    print("Datafile not exist.")
    exit()

model = tensorflow.keras.models.load_model(ModelFile)

img = img_to_array(load_img(Datafile, target_size=(INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNEL)))
# arrayに変換
img = np.array(img)
img = img.reshape(1, INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNEL)
# 画素値を0から1の範囲に変換
img = img.astype('float32')
detect_data = img / 255.0

detect_pred = model.predict(detect_data)

detect_score = np.mean(np.square(detect_data - detect_pred), axis=1)
detect_score = np.average(detect_score)
print('Score: ', detect_score)
