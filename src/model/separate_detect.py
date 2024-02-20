from pyexpat import model
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

from typing import List

from tensorflow.keras import Sequential, layers
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GRU,
    BatchNormalization,
    LSTM,
    InputLayer,
    Dropout,
    Flatten,
    Bidirectional,
    SimpleRNN,
    TimeDistributed,
    GlobalAveragePooling1D,
    Conv2D,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Reshape,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from data.data_labeling import DataLabeling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from feature.feature_extractor import FeatureExtractor


from model.base_model import BaseModel
from data.rhythm_detection import RhythmDetection
from data.data_processing import DataProcessing
from feature.audio_to_feature import AudioToFeature
from constant import (
    CHUNK_TIME_LENGTH,
    CODE2DRUM,
    METHOD_DETECT,
    MEL_SPECTROGRAM,
    MILLISECOND,
    CHUNK_LENGTH,
    SAMPLE_RATE,
)
from tensorflow.keras.layers import (
    Dense,
    GRU,
)


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
        self.predict_standard = 0.5
        self.n_rows = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param["hop_length"]
        self.n_columns = self.feature_param["n_fft"] // 2 + 1
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        self.win_length = self.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
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
        # # ----------------------------------------------------------------------------------------
        # input_data = Input(
        #     shape=(self.x_train.shape[0], CHUNK_TIME_LENGTH, 128, 1)
        # )  # frame, height, width, channels

        # convolution1 = TimeDistributed(
        #     Conv2D(filters=64, kernel_size=(1, 1), activation="tanh")
        # )(input_data)
        # bn = TimeDistributed(BatchNormalization())(convolution1)
        # convolution1 = TimeDistributed(
        #     Conv2D(filters=64, kernel_size=(1, 1), activation="tanh")
        # )(bn)
        # bn = TimeDistributed(BatchNormalization())(convolution1)
        # pooling1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convolution1)

        # convolution2 = TimeDistributed(
        #     Conv2D(filters=128, kernel_size=(1, 1), activation="tanh")
        # )(pooling1)
        # bn = TimeDistributed(BatchNormalization())(convolution2)
        # convolution2 = TimeDistributed(
        #     Conv2D(filters=128, kernel_size=(1, 1), activation="tanh")
        # )(convolution2)
        # bn = TimeDistributed(BatchNormalization())(convolution1)
        # pooling2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(bn)

        # convolution3 = TimeDistributed(
        #     Conv2D(filters=256, kernel_size=(1, 1), activation="tanh")
        # )(pooling2)
        # bn = TimeDistributed(BatchNormalization())(convolution3)
        # convolution3 = TimeDistributed(
        #     Conv2D(filters=256, kernel_size=(1, 1), activation="tanh")
        # )(convolution3)
        # bn = TimeDistributed(BatchNormalization())(convolution3)
        # pooling3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(bn)

        # flatten = TimeDistributed(Flatten())(pooling3)

        # dense = TimeDistributed(Dense(24))(flatten)

        # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(dense)
        # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
        # y_pred = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

        # dense = TimeDistributed(Dense(4, name="dense"))(y_pred)
        # # ----------------------------------------------------------------------------------------
        input_layer = Input(shape=(CHUNK_TIME_LENGTH, 128))
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
        output_layer = Dense(4, activation="sigmoid")(lstm3)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

        # compile the self.model
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
        audio_feature = BaseModel.split_data(audio_feature, CHUNK_TIME_LENGTH)

        # -- predict 결과 -- (#, time, 4 feature)
        predict_data = self.model.predict(audio_feature)
        predict_data = predict_data.reshape((-1, 4))
        # -- 12s 씩 잘린 거 이어붙이기 -> 함수로 뽑을 예정
        result_dict = {
            "HH": [row[0] for row in predict_data],
            "ST": [row[1] for row in predict_data],
            "SD": [row[2] for row in predict_data],
            "KK": [row[3] for row in predict_data],
        }

        # -- 실제 label
        true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        )

        DataLabeling.show_label_dict_compare_plot(true_label, result_dict, 0, 1200)

        # # -- get onsets
        # onsets_arr, drum_instrument = self.get_predict_onsets_instrument(predict_data)

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
