import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from onset_detection import OnsetDetect
import constant
import drum_cnn_model

onsetDetect = OnsetDetect(constant.SAMPLE_RATE, constant.ONSET_DURATION)
predict_model = tf.keras.models.load_model(constant.checkpoint_path)

"""
-- 전체 wav 주어졌을 때, 한 마디에 대한 rhythm 계산
"""
def get_bar_rhythm(audio_wav, bpm):
    return onsetDetect.get_rhythm(audio_wav, bpm, is_our_train_data=True)

"""
-- input  : onset마다 예측한 악기 확률
-- output : 일정 확률 이상으로 예측된 악기 추출
            [몇 번째 onset, [악기]]
            ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
"""
def get_predict_result(predict):
    # 각 행에서 threshold를 넘는 값의 인덱스 찾기
    indices_above_threshold = np.argwhere(predict > constant.predict_standard)

    current_row = indices_above_threshold[0, 0]
    result = []
    cols = []
    for index in indices_above_threshold:
        row, col = index
        if row != current_row:
            tmp = [row, cols]
            result.append(tmp)
            current_row = row
            cols = []
        cols.append(col)
    return result

"""
-- input  : 1 wav
-- output : 각 onset에 대한 악기 종류 분류
"""
def get_drum_instrument(audio):
    # -- trimmed audio
    trimmed_audios = drum_cnn_model.get_trimmed_audios(audio)
    
    # -- trimmed feature
    predict_data = []
    for _, taudio in enumerate(trimmed_audios):
        # -- mfcc
        trimmed_feature = drum_cnn_model.extract_audio_feature(taudio)
        predict_data.append(trimmed_feature)

    # -- reshape
    predict_data = drum_cnn_model.input_reshape(predict_data)
    # -- predict
    predict_data = predict_model.predict(predict_data)

    return get_predict_result(predict_data)

"""
-- 서버에서 호출할 함수
-- input  : 1 wav
-- output : {'instrument': [[1, [1, 7]], [2, [1]], ...], 'rhythm': [[0.0158, 0.054, ...], [0.0158, 0.054, ...], []]}
"""
def get_drum_data(wav_path, bpm):
    audio, _ = librosa.load(wav_path, sr=constant.SAMPLE_RATE, res_type='kaiser_fast')
    
    drum_instrument = get_drum_instrument(audio)
    bar_rhythm = get_bar_rhythm(audio, bpm)
    
    return {'instrument':drum_instrument, 'rhythm':bar_rhythm}

def main():
    wav_path='../../data/test_raw_data/P2_16_0001.m4a'
    bpm = 100
    result = get_drum_data(wav_path, bpm)
    print(result)

if __name__ == "__main__":
    main()