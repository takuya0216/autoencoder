# -*- coding: utf-8 -*-
# 実行方法　python3 this.py Datafoler

from copy import deepcopy
import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow.keras.models
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# 入力データサイズ
INPUT_SIZE = 64
IMAGE_CHANNEL = 3
BATCH_SIZE = 2

#確認画像出力フォルダ
CHECKIMG_PATH = "./image_check/"

#保存用フォルダ作成
if not os.path.isdir(CHECKIMG_PATH):
    os.makedirs(CHECKIMG_PATH)

ModelFile = "./ae_image_model.h5"

# 学習用データの読み込み
args = sys.argv
Datafolder = args[1]
if not os.path.exists(Datafolder):
    print("Datafolder not exist.")
    exit()

# フォルダ配下の画像を全て読み込む
IMAGES = []
Datafiles = Datafolder + '/*.jpg'
for pic in glob.glob(Datafiles):
    img = img_to_array(load_img(pic, target_size=(INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNEL)))
    IMAGES.append(img)
# arrayに変換
IMAGES = np.array(IMAGES)
# 画素値を0から1の範囲に変換
IMAGES = IMAGES.astype('float32')
train_data = IMAGES / 255.0
print('Number of Images: ', len(train_data))

#確認用に1つ除外
x_check = deepcopy(train_data[:1])
train_data = train_data[1:]

# AutoEncoder + Convolution モデルの組み立て
model = Sequential()
model.add(Conv2D(int(INPUT_SIZE/2), (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(int(INPUT_SIZE/4), (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(int(INPUT_SIZE/4), (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(int(INPUT_SIZE/2), (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(IMAGE_CHANNEL, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.build((BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNEL))
model.summary()

#学習中のログ
#======================================
class EpisodeLogger(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):

        #一定epoch毎に画像表示
        if epoch % 10 == 0:
            #======================================
            x_ans = model.predict(x_check)

            stack1 = np.concatenate(x_check, axis=1)
            stack2 = np.concatenate(x_ans, axis=1)
            stack3 = np.concatenate([stack1, stack2], axis=0)
            img = Image.fromarray(np.uint8(stack3 * 255.0))

            #plt.figure(figsize=(10, 10))
            img.save(os.path.join(CHECKIMG_PATH , 'check_' + str(INPUT_SIZE) + '_' + str(BATCH_SIZE) + '_' + str(epoch) + '.png'))
            #plt.imshow(img, vmin = 0, vmax = 255)
            #plt.show()
            #======================================

#======================================


# 学習：入力データと結果データを同一にする
history = model.fit(x=train_data, y=train_data,
                             epochs=200,
                             batch_size=BATCH_SIZE,
                             validation_split=0.1,
                             callbacks=[EpisodeLogger()])


# 学習済みモデルを保存
model.save(ModelFile)
print('Model saved.')
