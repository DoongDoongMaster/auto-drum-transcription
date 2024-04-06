import numpy as np

from glob import glob
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Conv1D,
    Input,
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
    GRU,
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Dropout,
)
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import tensorflow_addons as tfa

from feature.feature_extractor import FeatureExtractor
from feature.audio_to_feature import AudioToFeature
from data.rhythm_detection import RhythmDetection
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from model.base_model import BaseModel
from constant import (
    CHUNK_TIME_LENGTH,
    DETECT_TYPES,
    DRUM2CODE,
    METHOD_DETECT,
    MEL_SPECTROGRAM,
    MILLISECOND,
    SAMPLE_RATE,
    DETECT_CODE2DRUM,
    TEST,
    TRAIN,
    VALIDATION,
)


class SeparateDetectBModel(BaseModel):
    def __init__(
        self,
        training_epochs=40,
        opt_learning_rate=0.001,
        batch_size=20,
        unit_number=16,
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_DETECT,
            feature_type=MEL_SPECTROGRAM,
            compile_mode=True,
        )
        self.unit_number = unit_number
        self.predict_standard = 0.5
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

    def create_model_dataset(self, X, y, split_type):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = BaseModel.split_x_data(X, CHUNK_TIME_LENGTH)
        y = BaseModel.split_data(y, CHUNK_TIME_LENGTH)

        self.split_dataset(X, y, split_type)

    def create(self):
        input_layer = Input(shape=(self.n_rows, self.n_columns, 1))

        # First Convolutional Block
        conv1_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(input_layer)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        pool1 = MaxPooling2D(pool_size=(1, 3))(conv1_2)

        # Second Convolutional Block
        conv2_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(pool1)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="tanh", padding="same"
        )(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        pool2 = MaxPooling2D(pool_size=(1, 3))(conv2_2)

        # Reshape for RNN
        reshape = Reshape((pool2.shape[1], pool2.shape[2] * pool2.shape[3]))(pool2)

        # BiGRU layers
        lstm1 = Bidirectional(LSTM(50, return_sequences=True))(reshape)
        lstm2 = Bidirectional(LSTM(50, return_sequences=True))(lstm1)
        lstm3 = Bidirectional(LSTM(50, return_sequences=True))(lstm2)

        # Output layer
        output_layer = Dense(self.n_classes, activation="sigmoid")(lstm3)

        # Model compilation
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        opt = RMSprop(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"]
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
          {"CC":[0.0, 0.01, 0.18], "SD":[0.43, 0.76, 0.77, 1.07], ...}
        - 일정 확률 이상으로 예측된 악기 추출
          [몇 번째 onset, [악기]]
          ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_onsets_instrument(self, predict_data) -> List[float]:
        # predict standard 이상일 때 1, else 0
        onsets_arr = []
        drum_instrument = []
        each_instrument_onsets_arr = predict_data

        for i in range(len(predict_data)):
            is_onset = (
                False  # predict standard 이상 (1) 인 j가 하나라도 있다면 onset으로 판단
            )
            drums = []
            for j in range(self.n_classes):
                if predict_data[i][j] > self.predict_standard:
                    is_onset = True
                    drums.append(j)
                    each_instrument_onsets_arr[i][j] = 1
                else:
                    each_instrument_onsets_arr[i][j] = 0

            if is_onset:
                onsets_arr.append(i * self.hop_length / SAMPLE_RATE)
                drum_instrument.append([len(onsets_arr), drums])

        return onsets_arr, drum_instrument, each_instrument_onsets_arr

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
        # # -- 12s 씩 잘린 거 이어붙이기 -> 함수로 뽑을 예정
        result_dict = BaseModel.transform_arr_to_dict(predict_data)

        # -- 실제 label (merge cc into oh)
        class_6_true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        )
        # -- OH - CH
        class_6_true_label_arr = BaseModel.transform_dict_to_arr(class_6_true_label)
        grouping_true_label_arr = BaseModel.grouping_label(
            class_6_true_label_arr, DETECT_TYPES
        )
        true_label = BaseModel.transform_arr_to_dict(grouping_true_label_arr)

        # -- 각 라벨마다 peak picking
        true_label = BaseModel.transform_peakpick_from_dict(true_label)
        result_dict = BaseModel.transform_peakpick_from_dict(result_dict)

        DataLabeling.show_label_dict_compare_plot(true_label, result_dict, 0, 1200)

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
