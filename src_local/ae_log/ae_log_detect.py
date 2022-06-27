# -*- coding: utf-8 -*-
# 実行方法　python3 this.py datafile

import os
import sys
import numpy as np
import pandas as pd
import keras.models
from sklearn.metrics import mean_squared_error

ModelFile = "./ae_log_model.h5"

if not os.path.exists(ModelFile):
    print("Modelfile not exist.")
    exit()

args = sys.argv
Datafile = args[1]
if not os.path.exists(Datafile):
    print("Datafile not exist.")
    exit()

model = keras.models.load_model(ModelFile)

# 検出用データを読み込んで誤差を計算
detect_data = pd.read_csv(Datafile, header=None)
detect_pred = model.predict(detect_data)
detect_score = mean_squared_error(detect_data, detect_pred)
print('Score: ', detect_score)
