import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from onset_detection import OnsetDetect
import constant
import drum_cnn_model

"""
-- 경로 지정
"""
# -- wav data path
input_file_path = '../../data'
# -- raw data path
root_path = input_file_path + '/test_raw_data'
# -- 저장될 onset detected drum data folder path -- 예측 후 지워질 거임
trim_path = input_file_path + '/test_trim_data'

def main():
    # drum_cnn_model.create_trim_data(root_path, trim_path)

    model = tf.keras.models.load_model(constant.save_model_path)
    model.load_weights(constant.predict_checkpoint_path)

    datas = os.listdir(trim_path)
    feature_list = []
    for d in datas:
        if d.endswith('.wav'):
            path = os.path.join(trim_path, d)
            # -- feature: mfcc
            feature = drum_cnn_model.extract_feature(path)
            feature_list.append(feature)

    feature_list = drum_cnn_model.input_reshape(feature_list)
    predict = model.predict(feature_list, batch_size=constant.batch_size)

    # 각 행에서 threshold를 넘는 값의 인덱스 찾기
    indices_above_threshold = np.argwhere(predict > constant.predict_standard)

    # 출력
    current_row = indices_above_threshold[0, 0]
    result = []
    cols = []
    for index in indices_above_threshold:
        row, col = index
        if row != current_row:
            tmp = [row, cols]
            result.append(tmp)
            # print(row, ' >> ', *cols)
            current_row = row
            cols = []
        cols.append(col)

    print(result)

if __name__ == "__main__":
    main()