from typing import List
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input
from tensorflow.keras.optimizers import Adam

from feature.feature_extractor import FeatureExtractor
from feature.audio_to_feature import AudioToFeature
from data.rhythm_detection import RhythmDetection
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from model.base_model import BaseModel
from constant import (
    CHUNK_TIME_LENGTH,
    DRUM2CODE,
    METHOD_DETECT,
    MEL_SPECTROGRAM,
    MILLISECOND,
    SAMPLE_RATE,
    CODE2DRUM,
    CLASSIFY_DETECT_TYPES,
    CLASSIFY_CODE2DRUM,
)


def merge_columns(arr, col1, col2):
    # merge col2 into col1
    # -- 둘 중 하나라도 1이면 1
    # -- else, 둘 중 하나라도 0.5이면 0.5
    # -- else, 0
    merged_column = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        if 1 in arr[i, [col1, col2]]:
            merged_column[i] = 1
        elif 0.5 in arr[i, [col1, col2]]:
            merged_column[i] = 0.5
        else:
            merged_column[i] = 0

    # merge한 배열 col1 자리에 끼워넣기
    result = np.delete(arr, [col1, col2], axis=1)
    result = np.insert(result, col1, merged_column, axis=1)

    return result


class SeparateDetectModel(BaseModel):
    def __init__(
        self, training_epochs=40, opt_learning_rate=0.001, batch_size=20, unit_number=16
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_DETECT,
            feature_type=MEL_SPECTROGRAM,
        )
        self.unit_number = unit_number
        self.predict_standard = 0.4
        self.n_rows = CHUNK_TIME_LENGTH
        self.n_columns = self.feature_param["n_mels"]
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        self.win_length = self.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        return data

    def input_label_reshape(self, data):
        return data

    def output_reshape(self, data):
        return tf.reshape(data, [-1, self.n_rows, self.n_classes])

    def create_dataset(self):
        """
        -- load data from data file
        -- Implement dataset split feature & label logic
        """
        # Implement dataset split feature & label logic
        feature_df = FeatureExtractor.load_feature_file(
            self.method_type, self.feature_type, self.feature_extension
        )

        # -- get X, y
        X, y = BaseModel._get_x_y(self.method_type, feature_df)
        del feature_df

        # ------------------------------------------------------------------
        # y: CC, OH 합치기
        # l = {}
        # for k, v in CLASSIFY_DETECT_TYPES.items():
        #     temp_label = []
        #     for drum_idx, origin_key in enumerate(v):
        #         if len(temp_label) == 0:  # 초기화
        #             temp_label = y[CLASSIFY_DETECT_TYPES[k][drum_idx]]
        #         else:
        #             for frame_idx, frame_value in enumerate(y[origin_key]):
        #                 temp_label[frame_idx] = frame_value
        #     l[k] = temp_label
        # label_df = FeatureExtractor._make_label_dataframe(METHOD_DETECT, l)
        # y = label_df.to_numpy()
        # CC열과 OH열을 합치고 싶은 경우
        col1 = DRUM2CODE["CC"]
        col2 = DRUM2CODE["OH"]
        result = merge_columns(y, col1, col2)
        y = result
        del result

        # ------------------------------------------------------------------
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = BaseModel.split_x_data(X, CHUNK_TIME_LENGTH)
        y = BaseModel.split_data(y, CHUNK_TIME_LENGTH)

        # -- split train, val, test
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        del X
        del y

        x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
            x_train_temp,
            y_train_temp,
            test_size=0.2,
            random_state=42,
        )
        del x_train_temp
        del y_train_temp

        self.x_train = x_train_final
        self.x_val = x_val_final
        self.x_test = x_test
        self.y_train = y_train_final
        self.y_val = y_val_final
        self.y_test = y_test

        # -- print shape
        self.print_dataset_shape()

    def create(self):
        input_layer = Input(shape=(self.n_rows, self.n_columns))
        conv1 = Conv1D(
            filters=32, kernel_size=8, strides=1, activation="tanh", padding="same"
        )(input_layer)
        conv2 = Conv1D(
            filters=32, kernel_size=8, strides=1, activation="tanh", padding="same"
        )(conv1)
        conv3 = Conv1D(
            filters=32, kernel_size=8, strides=1, activation="tanh", padding="same"
        )(conv2)
        lstm1 = LSTM(32, return_sequences=True)(conv3)
        lstm2 = LSTM(32, return_sequences=True)(lstm1)
        lstm3 = LSTM(32, return_sequences=True)(lstm2)

        output_layer = Dense(self.n_classes, activation="sigmoid")(lstm3)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        opt = Adam(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    """
    -- 전체 wav 주어졌을 때, 한 마디에 대한 rhythm 계산
    """

    def get_bar_rhythm(self, audio_wav, bpm, onsets_arr):
        return RhythmDetection.get_rhythm(audio_wav, bpm, onsets_arr)

    """
    -- input  : time stamp마다 onset 확률 (모델 결과)
    -- output : 
        - onsets 배열
        - 일정 확률 이상으로 예측된 악기 추출
          [몇 번째 onset, [악기]]
          ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_onsets_instrument(self, predict_data) -> List[float]:
        onsets_arr = []
        drum_instrument = []

        for i in range(len(predict_data)):
            is_onset = (
                False  # predict standard 이상 (1) 인 j가 하나라도 있다면 onset으로 판단
            )
            drums = []
            for j in range(self.n_classes):
                if predict_data[i][j] > self.predict_standard:
                    is_onset = True
                    drums.append(j)
            if is_onset:
                onsets_arr.append(i * self.hop_length / SAMPLE_RATE)
                drum_instrument.append([len(onsets_arr), drums])

        return onsets_arr, drum_instrument

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio = FeatureExtractor.load_audio(wav_path)

        # -- cut delay
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)
        audio = new_audio

        # ------------------- compare predict with true label --------------------------
        audio_feature = np.zeros((0, 128))

        # 12s chunk하면서 audio feature추출 후 이어붙이기 -> 함수로 뽑을 예정
        audios = DataProcessing.cut_chunk_audio(audio)
        for i, ao in enumerate(audios):
            # audio to feature
            feature = AudioToFeature.extract_feature(
                ao, self.method_type, self.feature_type
            )
            audio_feature = np.vstack([audio_feature, feature])

        scaler = StandardScaler()
        audio_feature = scaler.fit_transform(audio_feature)

        # -- input (#, time, 128 feature)
        audio_feature = BaseModel.split_x_data(audio_feature, CHUNK_TIME_LENGTH)

        # -- predict 결과 -- (#, time, 4 feature)
        predict_data = self.model.predict(audio_feature)
        predict_data = predict_data.reshape((-1, self.n_classes))
        # -- 12s 씩 잘린 거 이어붙이기 -> 함수로 뽑을 예정
        result_dict = {}
        for code, drum in CLASSIFY_CODE2DRUM.items():
            result_dict[drum] = [row[code] for row in predict_data]

        # -- 실제 label
        # true_label = DataLabeling.data_labeling(
        #     audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        # )
        # -- 원래 정답 라벨
        tmp_true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        )
        true_label = {}
        for k, v in CLASSIFY_DETECT_TYPES.items():
            temp_label = []
            for drum_idx, origin_key in enumerate(v):
                if len(temp_label) == 0:  # 초기화
                    temp_label = tmp_true_label[CLASSIFY_DETECT_TYPES[k][drum_idx]]
                else:
                    for frame_idx, frame_value in enumerate(tmp_true_label[origin_key]):
                        temp_label[frame_idx] = frame_value
            true_label[k] = temp_label

        DataLabeling.show_label_dict_compare_plot(true_label, result_dict, 0, 1200)

        # # -- get onsets
        # onsets_arr, drum_instrument = self.get_predict_onsets_instrument(predict_data)

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
