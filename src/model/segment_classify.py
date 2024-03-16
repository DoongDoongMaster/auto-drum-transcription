import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Dense, LSTM, Conv1D, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import BinaryAccuracy

from model.base_model import BaseModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from data.rhythm_detection import RhythmDetection
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from constant import (
    CLASSIFY_DETECT_TYPES,
    METHOD_CLASSIFY,
    MFCC,
    MILLISECOND,
    SAMPLE_RATE,
    CLASSIFY_DURATION,
    PKL,
    CLASSIFY_CODE2DRUM,
)


class SegmentClassifyModel(BaseModel):
    def __init__(
        self,
        training_epochs=40,
        opt_learning_rate=0.001,
        batch_size=20,
        feature_type=MFCC,
        feature_extension=PKL,
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_CLASSIFY,
            feature_type=feature_type,
            feature_extension=feature_extension,
        )
        self.data_cnt = 2
        self.train_cnt = 1
        self.predict_standard = 0.8
        self.n_rows = (
            self.feature_param["n_mfcc"]
            if feature_type == MFCC
            else self.feature_param["n_mels"]
        )
        self.n_columns = (
            int(CLASSIFY_DURATION * SAMPLE_RATE) // self.feature_param["hop_length"]
        )
        self.n_channels = (
            self.feature_param["n_channels"] if feature_type == MFCC else 1
        )
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        # self.load_model("../models/classify_mfcc_2024-02-24_00-53-10_smote_5.h5")
        self.load_model()

    def input_reshape(self, data):
        # cnn data
        # return tf.reshape(
        #     data,
        #     [
        #         -1,
        #         self.n_rows,
        #         self.n_columns,
        #         self.n_channels,
        #     ],
        # )
        # sequence data
        return tf.reshape(
            data,
            [
                -1,
                self.n_columns,
                self.n_rows,
            ],
        )

    def x_data_1d_reshape(self, data):
        return tf.reshape(
            data,
            [
                -1,
                self.n_rows * self.n_columns * self.n_channels,
            ],
        )

    def x_data_2d_reshape(self, data):
        return tf.reshape(data, [-1, self.n_columns])

    @staticmethod
    def x_data_transpose(data):
        """
        -- 시계열 모델 학습 시, 데이터 transpose (1차원: 데이터 개수, 2차원: time stamp, 3차원: feature 개수)
        """
        return np.transpose(data, (0, 2, 1))

    @staticmethod
    def delete_small_data(counter_y, X, number_y):
        """
        -- SMOTE 전에 데이터 분포에서 너무 적은 개수를 가진 데이터는 삭제하는 함수
        """
        SMALL_STANDARD = 300

        # 데이터 적은 라벨 번호 구하기
        small_label = []
        for key, value in counter_y.items():
            if value < SMALL_STANDARD:
                small_label.append(key)

        # 데이터 적은 라벨을 지닌 데이터 인덱스 구하기
        small_y = np.array([])
        number_y = number_y.ravel()
        for l in small_label:
            small_y = np.append(small_y, np.where(number_y == l))

        # 데이터 삭제
        small_y = small_y.astype(int)
        new_x = np.delete(X, small_y, axis=0)
        new_y = np.delete(number_y, small_y, axis=0)
        return new_x, new_y

    @staticmethod
    def smote_data(x_1d, number_y):
        smt = SMOTE(random_state=42)

        x_1d, number_y = smt.fit_resample(x_1d, number_y)

        # 비율 확인
        counter = Counter(number_y)
        print("변경 후", counter)

        return x_1d, number_y

    def load_dataset(self, feature_files: list[str] = None):
        """
        -- load data from data file
        """
        # Implement dataset split feature & label logic
        feature_df = FeatureExtractor.load_feature_file(
            self.method_type, self.feature_type, self.feature_extension, feature_files
        )

        # -- get X, y
        X, y = BaseModel._get_x_y(self.method_type, feature_df)
        del feature_df

        X = SegmentClassifyModel.x_data_transpose(X)

        number_y = FeatureExtractor.one_hot_label_to_number(y)
        counter = Counter(number_y)
        print("변경 전", counter)

        label_cnt = {}  # label별 나눠서 학습시킬 데이터 개수
        total = 0
        for label, cnt in counter.items():
            label_cnt[label] = cnt // self.train_cnt
            total += label_cnt[label]

        label_temp_cnt = {l: 0 for l in counter.keys()}  # 각 라벨별 개수
        label_idx = {l: 0 for l in counter.keys()}  # 각 라벨별 인덱스
        split_data = [
            {"x": [], "y": []} for _ in range(self.train_cnt)
        ]  # 나눈 데이터 형태

        for idx, label in enumerate(number_y):  # 각 라벨별로 label_cnt 개수만큼 나누기
            split_data[label_idx[label]]["x"].append(X[idx])
            split_data[label_idx[label]]["y"].append(label)
            label_temp_cnt[label] += 1
            if (
                label_temp_cnt[label] == label_cnt[label]
                and label_idx[label] < self.train_cnt - 1
            ):
                label_temp_cnt[label] = 0
                label_idx[label] += 1

        return split_data

    def create_dataset(self, X, y):
        """
        -- Implement dataset split feature & label logic
        """
        # X = self.x_data_1d_reshape(X)
        # X, y = SegmentClassifyModel.smote_data(X, y)

        y = FeatureExtractor.number_to_one_hot_label(y)

        # -- split train, val, test
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        del X
        del y

        x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
            x_train_temp,
            y_train_temp,
            test_size=0.2,
            random_state=42,
            stratify=y_train_temp,
        )
        del x_train_temp
        del y_train_temp

        # input shape 조정
        x_train_final = self.input_reshape(x_train_final)
        x_val_final = self.input_reshape(x_val_final)
        x_test = self.input_reshape(x_test)

        self.x_train = x_train_final
        self.x_val = x_val_final
        self.x_test = x_test
        self.y_train = y_train_final
        self.y_val = y_val_final
        self.y_test = y_test

        # -- print shape
        self.print_dataset_shape()

    def create(self):
        n_steps = self.n_columns
        n_features = self.n_rows

        keras.backend.clear_session()

        # self.model = keras.Sequential(
        #     [
        #         layers.Input(shape=(n_steps, n_features)),
        #         layers.Conv1D(
        #             filters=64,
        #             kernel_size=8,
        #             padding="same",
        #             data_format="channels_last",
        #             dilation_rate=1,
        #             activation="relu",
        #         ),
        #         layers.LSTM(
        #             units=32, activation="tanh", name="lstm_1", return_sequences=True
        #         ),
        #         layers.Dropout(0.2),
        #         layers.LSTM(
        #             units=32, activation="tanh", name="lstm_2", return_sequences=True
        #         ),
        #         layers.Dropout(0.2),
        #         layers.Flatten(),
        #         layers.Dense(self.n_classes, activation="sigmoid"),
        #     ]
        # )

        self.model = keras.Sequential(
            [
                layers.Input(shape=(n_steps, n_features)),
                layers.Conv1D(
                    filters=64,
                    kernel_size=8,
                    padding="same",
                    data_format="channels_last",
                    dilation_rate=1,
                    activation="tanh",
                ),
                layers.Bidirectional(
                    layers.LSTM(64, activation="tanh", return_sequences=True)
                ),
                layers.Dropout(0.2),
                layers.Bidirectional(
                    layers.LSTM(32, activation="tanh", return_sequences=True)
                ),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(self.n_classes, activation="sigmoid"),
            ]
        )
        self.model.summary()
        # compile the self.model
        opt = Adam(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=[BinaryAccuracy(threshold=self.predict_standard)],
        )

    def run(self):
        """
        데이터셋 생성, 모델 생성, 학습, 평가, 모델 저장 파이프라인
        """
        save_folder_path = FeatureExtractor._get_save_folder_path(
            self.method_type, self.feature_type
        )
        feature_files = glob(f"{save_folder_path}/*.{self.feature_extension}")
        feature_file_offset = math.ceil(len(feature_files) / float(self.data_cnt))
        for i in range(self.data_cnt):
            split_dataset = self.load_dataset(
                feature_files[i * feature_file_offset : (i + 1) * feature_file_offset]
            )
            self.create()

            for data in split_dataset:
                print("split data length", len(data["x"]))
                self.create_dataset(data["x"], data["y"])
                self.train()
                self.evaluate()
        self.save()

    """
    -- 전체 wav 주어졌을 때, 한 마디에 대한 rhythm 계산
    """

    def get_bar_rhythm(self, audio_wav, bpm, onsets_arr):
        return RhythmDetection.get_rhythm(audio_wav, bpm, onsets_arr)

    """
    -- input  : onset마다 예측한 악기 확률
    -- output : 일정 확률 이상으로 예측된 악기 추출
                [몇 번째 onset, [악기]]
                ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_result(self, predict):
        # 각 행에서 threshold를 넘는 값의 인덱스 찾기
        indices_above_threshold = np.argwhere(predict > self.predict_standard)

        if indices_above_threshold.size == 0:
            raise Exception("no predict data")

        current_row = indices_above_threshold[0, 0]
        result = []
        cols = []
        for index in indices_above_threshold:
            row, col = index
            if row != current_row:
                tmp = [current_row, cols]
                result.append(tmp)
                current_row = row
                cols = []
            cols.append(col)
        result.append([current_row, cols])
        return result

    """
    -- input  : 1 wav
    -- output : 각 onset에 대한 악기 종류 분류
    """

    def get_drum_instrument(self, audio, bpm):
        # -- trimmed audio
        onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)
        trimmed_audios = DataProcessing.trim_audio_per_onset(audio, onsets_arr)

        # -- trimmed feature
        predict_data = []
        for _, taudio in enumerate(trimmed_audios):
            trimmed_feature = AudioToFeature.extract_feature(
                taudio, self.method_type, self.feature_type
            )
            predict_data.append(trimmed_feature)

        # # standard scaler
        # predict_data = self.x_data_1d_reshape(predict_data)
        # scaler = StandardScaler()
        # predict_data = scaler.fit_transform(predict_data)

        # -- reshape
        predict_data = SegmentClassifyModel.x_data_transpose(predict_data)

        # -- predict
        predict_data = self.model.predict(predict_data)

        print("-- ! classify 방법 예측 결과 ! --")
        np.set_printoptions(precision=2, suppress=True)
        print(predict_data)

        return self.get_predict_result(predict_data)

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio = FeatureExtractor.load_audio(wav_path)

        # -- instrument
        drum_instrument = self.get_drum_instrument(audio, bpm)
        # -- rhythm
        onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)

        # -- 원래 정답 라벨
        true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_CLASSIFY, hop_length=self.hop_length
        )
        l = {}
        for k, v in CLASSIFY_DETECT_TYPES.items():
            temp_label = []
            for drum_idx, origin_key in enumerate(v):
                if len(temp_label) == 0:  # 초기화
                    temp_label = true_label[CLASSIFY_DETECT_TYPES[k][drum_idx]]
                else:
                    for frame_idx, frame_value in enumerate(true_label[origin_key]):
                        if temp_label[frame_idx] == 1.0 or frame_value == 0.0:
                            continue
                        temp_label[frame_idx] = frame_value
            l[k] = temp_label
        # print(l)

        # DataLabeling.show_label_dict_plot(true_label)

        # -- transport frame
        onset_dict = {v: [] for _, v in CLASSIFY_CODE2DRUM.items()}
        for data in drum_instrument:
            idx = data[0]
            instrument = data[1]
            for inst in instrument:
                onset_dict[CLASSIFY_CODE2DRUM[inst]].append(onsets_arr[idx])
        frame_length = len(audio) // self.hop_length
        frame_onset = DataLabeling._get_label_detect(
            onset_dict, frame_length, self.hop_length
        )
        new_frame_onset = {}
        for k, v in frame_onset.items():
            if k in list(CLASSIFY_CODE2DRUM.values()):
                new_frame_onset[k] = v

        DataLabeling.show_label_dict_compare_plot(l, new_frame_onset, 0, 2400)
        # DataLabeling.show_label_dict_plot(new_frame_onset)

        # delay 제거
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        return {"instrument": drum_instrument, "rhythm": bar_rhythm}
