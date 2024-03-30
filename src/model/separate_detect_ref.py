from typing import List
import librosa
import numpy as np

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
from tensorflow.keras.optimizers import RMSprop

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
    DETECT_CODE2DRUM,
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
    result = result.flatten()

    return result


class SeparateDetectRefModel(BaseModel):
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
        # y: 0 CC, 1 OH, 2 CH 합치기
        col2 = DRUM2CODE["CH"]
        col1 = DRUM2CODE["OH"]
        col0 = DRUM2CODE["CC"]
        result = merge_columns(y, col1, col2)
        result = merge_columns(result, col0, col1)
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
        input_layer = Input(shape=(self.n_rows, self.n_columns, 1))

        # 1st Convolutional Block
        conv1_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(input_layer)
        conv1_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(conv1_1)
        pool1 = MaxPooling2D(pool_size=(1, 3))(conv1_2)
        dropout1 = Dropout(0.4)(pool1)

        # 2nd Convolutional Block
        conv2_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout1)
        conv2_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(conv2_1)
        pool2 = MaxPooling2D(pool_size=(1, 3))(conv2_2)
        dropout2 = Dropout(0.4)(pool2)

        # 3rd Convolutional Block
        conv3_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout2)
        conv3_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(conv3_1)
        pool3 = MaxPooling2D(pool_size=(1, 3))(conv3_2)
        dropout3 = Dropout(0.4)(pool3)

        # 4th Convolutional Block
        conv4_1 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout3)
        conv4_2 = Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(conv4_1)
        pool4 = MaxPooling2D(pool_size=(1, 3))(conv4_2)
        dropout4 = Dropout(0.1)(pool4)

        # Reshape for RNN
        reshape = Reshape((dropout4.shape[1], dropout4.shape[2] * dropout4.shape[3]))(
            dropout4
        )

        # BiLSTM layers
        lstm1 = Bidirectional(LSTM(50, return_sequences=True))(reshape)
        lstm2 = Bidirectional(LSTM(50, return_sequences=True))(lstm1)
        lstm3 = Bidirectional(LSTM(50, return_sequences=True))(lstm2)
        last_dropout = Dropout(0.4)(lstm3)

        # Output layer
        output_layer = Dense(self.n_classes, activation="sigmoid")(last_dropout)

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

    # tranform 2D array to dict
    def transform_arr_to_dict(self, arr_data):
        result_dict = {}
        for code, drum in DETECT_CODE2DRUM.items():
            result_dict[drum] = [row[code] for row in arr_data]
        return result_dict

    def transform_peakpick_from_dict(self, data_dict):
        result_dict = {}
        for key, values in data_dict.items():
            item_value = np.array(values)
            peak_value = np.zeros(len(item_value))

            # peak_pick를 통해 몇 번째 인덱스가 peak인지 추출
            peaks = librosa.util.peak_pick(
                item_value,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=3,
                delta=0.1,
                wait=1,
            )
            for idx in peaks:
                peak_value[idx] = 1
            result_dict[key] = peak_value
        return result_dict

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
        result_dict = self.transform_arr_to_dict(predict_data)

        # -- threshold 0.5
        onsets_arr, drum_instrument, each_instrument_onsets_arr = (
            self.get_predict_onsets_instrument(predict_data)
        )
        threshold_dict = self.transform_arr_to_dict(each_instrument_onsets_arr)

        # predict : threshold 둘만 비교
        # DataLabeling.show_pred_dict_plot_detect(result_dict, threshold_dict, 0, 1200)

        # -- 실제 label (merge cc into oh)
        class_6_true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        )
        # -- OH - CH
        keys_to_extract = ["OH", "CH"]
        selected_values = [class_6_true_label[key] for key in keys_to_extract]
        oh_ch_label = np.vstack(selected_values).T
        merged_cc_oh = merge_columns(oh_ch_label, 0, 1)
        class_6_true_label.pop("CH", None)
        class_6_true_label["OH"] = merged_cc_oh.flatten()
        class_5_true_label = class_6_true_label  # -- class 5
        # -- CC - OH
        keys_to_extract_s = ["CC", "OH"]
        selected_values_s = [class_5_true_label[key] for key in keys_to_extract_s]
        cc_oh_label = np.vstack(selected_values_s).T
        merged_cc_oh = merge_columns(cc_oh_label, 0, 1)
        class_5_true_label.pop("CC", None)
        class_5_true_label["OH"] = merged_cc_oh
        class_4_true_label = class_5_true_label  # -- class 4

        true_label = class_4_true_label

        # -- 각 라벨마다 peak picking
        true_label = self.transform_peakpick_from_dict(true_label)
        result_dict = self.transform_peakpick_from_dict(result_dict)
        threshold_dict = self.transform_peakpick_from_dict(threshold_dict)

        DataLabeling.show_label_dict_compare_plot_detect(
            true_label, result_dict, threshold_dict, 0, 1200
        )

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
