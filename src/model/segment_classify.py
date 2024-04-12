import math
import numpy as np
from numpy.core.multiarray import array as array
import tensorflow as tf

from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from imblearn.over_sampling import SMOTE
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
    CLASSIFY_TYPES,
    DRUM2CODE,
    ENST,
    IDMT,
    METHOD_CLASSIFY,
    MFCC,
    MILLISECOND,
    SAMPLE_RATE,
    CLASSIFY_DURATION,
    PKL,
    CLASSIFY_CODE2DRUM,
    LABEL_DDM,
    TEST,
    TRAIN,
    VALIDATION,
)


class SegmentClassifyModel(BaseModel):
    def __init__(
        self,
        training_epochs=40,
        opt_learning_rate=0.001,
        batch_size=20,
        feature_type=MFCC,
        feature_extension=PKL,
        load_model_flag=True,
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_CLASSIFY,
            feature_type=feature_type,
            feature_extension=feature_extension,
            class_dict=CLASSIFY_CODE2DRUM,
        )
        self.data_cnt = 1
        self.train_cnt = 1
        self.predict_standard = 0.5
        self.n_rows = (
            self.feature_param["n_mfcc"]
            if feature_type == MFCC
            else self.feature_param["n_mels"]
        )  # feature 개수
        self.n_columns = (
            int(CLASSIFY_DURATION * SAMPLE_RATE) // self.feature_param["hop_length"]
        )  # timestamp
        self.n_channels = 1
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        if load_model_flag:
            self.load_model()

    def input_reshape(self, data):
        print("=======origin shape======", data.shape)
        data = data[:, : self.n_columns, :]
        # sequence data
        return np.reshape(
            data,
            [
                -1,
                self.n_columns,
                self.n_rows,
                self.n_channels,
            ],
        )

    def x_data_1d_reshape(self, data):
        return np.reshape(
            data,
            [
                -1,
                self.n_rows * self.n_columns * self.n_channels,
            ],
        )

    def x_data_2d_reshape(self, data):
        return np.reshape(data, [-1, self.n_columns])

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
        y = BaseModel.grouping_label(y, CLASSIFY_TYPES)

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

    def create_model_dataset(self, X: np.array, y: np.array, split_type: str):
        X = SegmentClassifyModel.x_data_transpose(X)

        if split_type == TRAIN:
            x_train, x_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
            del X, y

            x_val = self.input_reshape(x_val)
            self.split_dataset(x_val, y_val, VALIDATION)
            del x_val, y_val

            # train data smote
            # x_train = self.x_data_1d_reshape(x_train)
            y_train = FeatureExtractor.one_hot_label_to_number(y_train)

            # 라벨 비율 확인
            counter = Counter(y_train)
            print("변경 전", counter)

            # x_train, y_train = SegmentClassifyModel.smote_data(x_train, y_train)
            # decimal to multi-hot-encoding
            y = FeatureExtractor.number_to_one_hot_label(y_train)

            # input shape 조정
            x_train = self.input_reshape(x_train)
            self.split_dataset(x_train, y_train, split_type)

            del x_train, y_train
        elif split_type == TEST:
            # input shape 조정
            x_test = self.input_reshape(X)
            self.split_dataset(x_test, y, split_type)

            del X, y

    def create(self):
        n_steps = self.n_columns
        n_features = self.n_rows

        keras.backend.clear_session()

        input_layer = Input(shape=(n_steps, n_features, self.n_channels))

        # 1st Convolutional Block
        conv1_1 = layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation="selu", padding="same"
        )(input_layer)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1_1)
        dropout1 = layers.Dropout(0.1)(pool1)

        # 2st Convolutional Block
        conv2_1 = layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2_1)
        dropout2 = layers.Dropout(0.1)(pool2)

        # 3st Convolutional Block
        conv3_1 = layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3_1)
        dropout3 = layers.Dropout(0.1)(pool3)

        # 4st Convolutional Block
        conv4_1 = layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation="selu", padding="same"
        )(dropout3)
        pool4 = layers.MaxPooling2D(pool_size=(1, 2))(conv4_1)
        dropout4 = layers.Dropout(0.1)(pool4)

        # Reshape for RNN
        reshape = layers.Reshape(
            (dropout4.shape[1], dropout4.shape[2] * dropout4.shape[3])
        )(dropout4)

        # BiLSTM layers
        lstm1 = layers.Bidirectional(
            LSTM(32, return_sequences=True, activation="tanh")
        )(reshape)
        dropout_lstm1 = layers.Dropout(0.1)(lstm1)
        lstm2 = layers.Bidirectional(
            LSTM(32, return_sequences=True, activation="tanh")
        )(dropout_lstm1)
        dropout_lstm2 = layers.Dropout(0.1)(lstm2)
        last_flatten = layers.Flatten()(dropout_lstm2)

        # Output layer
        output_layer = Dense(self.n_classes, activation="sigmoid")(last_flatten)

        # model compile
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

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
        save_folder_path_train_enst = FeatureExtractor._get_save_folder_path(
            self.method_type, self.feature_type, ENST, "train"
        )
        save_folder_path_test_enst = FeatureExtractor._get_save_folder_path(
            self.method_type, self.feature_type, ENST, "test"
        )
        save_folder_path_train_idmt = FeatureExtractor._get_save_folder_path(
            self.method_type, self.feature_type, IDMT, "train"
        )
        save_folder_path_test = FeatureExtractor._get_save_folder_path(
            self.method_type, self.feature_type, IDMT, "test"
        )

        feature_files_train_enst = glob(
            f"{save_folder_path_train_enst}/*.{self.feature_extension}"
        )
        feature_files_test_enst = glob(
            f"{save_folder_path_test_enst}/*.{self.feature_extension}"
        )
        feature_files_train_idmt = glob(
            f"{save_folder_path_train_idmt}/*.{self.feature_extension}"
        )
        feature_files_train = (
            feature_files_train_enst
            + feature_files_test_enst
            + feature_files_train_idmt
        )
        feature_files_test = glob(f"{save_folder_path_test}/*.{self.feature_extension}")
        feature_file_offset_train = math.ceil(
            len(feature_files_train) / float(self.data_cnt)
        )
        feature_file_offset_test = math.ceil(
            len(feature_files_test) / float(self.data_cnt)
        )
        self.create()
        for i in range(self.data_cnt):
            split_dataset_train = self.load_dataset(
                feature_files_train[
                    i * feature_file_offset_train : (i + 1) * feature_file_offset_train
                ]
            )
            split_dataset_test = self.load_dataset(
                feature_files_test[
                    i * feature_file_offset_test : (i + 1) * feature_file_offset_test
                ]
            )
            for idx, train_data in enumerate(split_dataset_train):
                print("split data length", len(train_data["x"]))
                self.create_dataset(train_data, split_dataset_test[idx])
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
                tmp = [int(current_row), cols]
                result.append(tmp)
                current_row = row
                cols = []
            cols.append(int(col))
        result.append([int(current_row), cols])
        return result

    """
    -- input  : 1 wav
    -- output : 각 onset에 대한 악기 종류 분류
    """

    def get_drum_instrument(self, audio, bpm):
        # -- trimmed audio
        onsets_arr = OnsetDetect.onset_detection(audio)
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
        onsets_arr = OnsetDetect.onset_detection(audio)

        # -- 원래 정답 라벨
        true_label = DataLabeling.data_labeling(
            audio,
            wav_path,
            METHOD_CLASSIFY,
            hop_length=self.hop_length,
        )
        l = {}
        for k, v in CLASSIFY_TYPES.items():
            temp_label = []
            for drum_idx, origin_key in enumerate(v):
                if len(temp_label) == 0:  # 초기화
                    temp_label = true_label[CLASSIFY_TYPES[k][drum_idx]]
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
            onset_dict, frame_length, self.hop_length, LABEL_DDM
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

    # tranform 2D array to dict
    def transform_arr_to_dict(self, arr_data):
        result_dict = {}
        for code, drum in CLASSIFY_CODE2DRUM.items():
            result_dict[drum] = [row[code] for row in arr_data]
        return result_dict

    def data_pre_processing(self, audio: np.array) -> np.array:
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

        # -- reshape
        predict_data = SegmentClassifyModel.x_data_transpose(predict_data)
        return predict_data

    def data_post_processing(
        self, predict_data: np.array, audio: np.array, label_cnt: int = 4
    ):
        predict_data_result = []
        if label_cnt == 3:
            predict_data_result = np.insert(
                predict_data, 1, np.zeros((1, len(predict_data))), axis=1
            )
        elif label_cnt == 5:
            for data in predict_data:
                data_0 = max(data[0], data[1])
                predict_data_result.append([data_0, data[2], data[3], data[4]])
        elif label_cnt == 4:
            predict_data_result = predict_data

        predict_data_result = np.array(predict_data_result)
        drum_instrument = self.get_predict_result(predict_data_result)
        onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)
        onsets_arr = onsets_arr.tolist()

        return drum_instrument, onsets_arr
