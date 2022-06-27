# -*- coding: utf-8 -*-

# 例）毎時00分にデータを検出にかける
# crontab -e
# 0 * * * * python3 /folder/this.py data.wav

import os
import sys
import numpy as np
import pandas as pd
import keras.models
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from sklearn.metrics import mean_squared_error

# 入力データのカラム数
INPUT_SIZE = 20

ModelFile = "./ae_audio_model.h5"

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
(rate,sig) = wav.read(Datafile)
detect_data = logfbank(sig,rate,winlen=0.01,nfilt=INPUT_SIZE)

detect_pred = model.predict(detect_data)
detect_score = mean_squared_error(detect_data, detect_pred)


##############################
# 検出結果が異常であればアクション
import smtplib
from email.mime.text import MIMEText

# 異常と判別するしきい値
BORDER = 1.0

if detect_pred > 1.0:
    # MIMETextを作成
    message = "異常を検出しました。"
    msg = MIMEText(message)
    msg["Subject"] = "AI検出アラート"
    msg["To"] = "送信先@aaa.com"
    msg["From"] = "送信元@bbb.com"

    # サーバを指定する
    server = smtplib.SMTP("smtp.bbb.com(社内SMTPサーバ)", 25(ポート番号))
    # メールを送信する
    server.send_message(msg)
    # 閉じる
    server.quit()
