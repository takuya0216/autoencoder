# -*- coding: utf-8 -*-
# 実行方法　python3 this.py datafile

import os
import sys
import numpy as np
import cv2
from tensorflow.keras import models
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import tensorflow as tf
from PIL import Image

# 入力データサイズ
INPUT_SIZE = 64
IMAGE_CHANNEL = 3
IMAGE_SIZE  = (INPUT_SIZE, INPUT_SIZE)

# grad-cam可視化ファイル保存先
HEATMAP_IMAGE_PATH = "./image_heatmap"

ModelFile = "./ae_image_model.h5"

if not os.path.exists(ModelFile):
    print("Modelfile not exist.")
    exit()

args = sys.argv
Datafile = args[1]
if not os.path.exists(Datafile):
    print("Datafile not exist.")
    exit()

model = tf.keras.models.load_model(ModelFile)

#モデルの最後の層の名前を確認
print([layer.name for layer in model.layers])

# grad_cam実装
def grad_cam(input_model, input_image, layer_name):
    """
    Args:
        input_model(object): モデルオブジェクト
        input_image(ndarray): 画像
        layer_name(string): 畳み込み層の名前
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """

    # 画像の前処理
    # 読み込む画像が1枚なため、次元を増やしておかないとmode.predictが出来ない
    input_image = np.expand_dims(input_image, axis=0)
    preprocessed_input = input_image.astype('float32') / 255.0

    grad_model = tf.keras.models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # 勾配を計算
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, IMAGE_SIZE, cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()

    # モノクロ画像に疑似的に色をつける
    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    # RGBに変換
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    # もとの画像に合成
    output_image = (np.float32(rgb_cam) + input_image / 2)

    return output_image


# 画像読み込み
img = img_to_array(load_img(Datafile, target_size=(INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNEL)))
grad_cam_img = img
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

# grad-camで可視化
target_layer = 'activation_4'
cam = grad_cam(model, grad_cam_img, target_layer)
cam = np.concatenate(cam, axis=1)
heatmap_img = Image.fromarray(np.uint8(cam * 255.0))
heatmap_img.save(os.path.join(HEATMAP_IMAGE_PATH , 'heatmap_' + str(INPUT_SIZE) + '.png'))
