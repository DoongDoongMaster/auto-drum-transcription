import librosa
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd

from glob import glob
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
)

from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from constant import (
    DETECT_CODE2DRUM,
    SAMPLE_RATE,
    PKL,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    FEATURE_PARAM,
    ROOT_PATH,
    RAW_PATH,
    CODE2DRUM,
    CLASSIFY_CODE2DRUM,
)


class BaseModel:
    def __init__(
        self,
        training_epochs,
        opt_learning_rate,
        batch_size,
        method_type,
        feature_type,
        feature_extension=PKL,
    ):
        self.model = None
        self.training_epochs = training_epochs
        self.opt_learning_rate = opt_learning_rate
        self.batch_size = batch_size
        self.method_type = method_type
        self.feature_type = feature_type
        self.feature_extension = feature_extension
        self.feature_param = FEATURE_PARAM[method_type][feature_type]
        self.sample_rate = SAMPLE_RATE
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_val: np.ndarray = None
        self.y_val: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.save_path = f"../models/{method_type}_{feature_type}"
        self.model_save_type = "h5"

    def save(self):
        """
        -- 학습한 모델 저장하기
        """
        # 현재 날짜와 시간 가져오기
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 모델 저장
        model_path = f"{self.save_path}_{date_time}.{self.model_save_type}"
        self.model.save(model_path)
        print("--! save model: ", model_path)

    def load_model(self, model_file=None):
        """
        -- method_type과 feature type에 맞는 가장 최근 모델 불러오기
        """
        model_files = glob(f"{self.save_path}_*.{self.model_save_type}")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        load_model_file = model_files[0]  # 가장 최근 모델

        if model_file is not None:  # 불러오고자 하는 특정 모델 파일이 있다면
            load_model_file = model_file

        print("-- ! load model: ", load_model_file)
        self.model = tf.keras.models.load_model(load_model_file)

    def input_reshape(self, data) -> np.ndarray:
        # Implement input reshaping logic
        pass

    @staticmethod
    def _get_x_y(method_type: str, feature_df: pd.DataFrame):
        if method_type == METHOD_CLASSIFY:
            X = np.array(feature_df.feature.tolist())
            y = feature_df[[drum for _, drum in CLASSIFY_CODE2DRUM.items()]].to_numpy()
            return X, y
        if method_type in METHOD_DETECT:
            # label(HH, ST, SD, KK onset 여부) | mel-1, mel-2, mel-3, ...
            X = feature_df.drop(
                [drum for _, drum in CODE2DRUM.items()], axis=1
            ).to_numpy()
            y = feature_df[[drum for _, drum in CODE2DRUM.items()]].to_numpy()
            return X, y
        if method_type in METHOD_RHYTHM:
            # label(onset 여부) | mel-1, mel-2, mel-3, ...
            X = feature_df.drop(["label"], axis=1).to_numpy()
            y = feature_df["label"].to_numpy()
            return X, y

    # 데이터 분할을 위한 함수 정의
    @staticmethod
    def split_data(data, chunk_size):
        num_samples, num_features = data.shape
        num_chunks = num_samples // chunk_size

        # 나머지 부분을 제외한 데이터만 사용
        data = data[: num_chunks * chunk_size, :]

        # reshape을 통해 3D 배열로 변환
        return data.reshape((num_chunks, chunk_size, num_features))
        # 데이터 분할을 위한 함수 정의

    @staticmethod
    def split_x_data(data, chunk_size):
        num_samples, num_features = data.shape
        num_chunks = num_samples // chunk_size

        # 나머지 부분을 제외한 데이터만 사용
        data = data[: num_chunks * chunk_size, :]

        # reshape을 통해 3D 배열로 변환
        return data.reshape((num_chunks, chunk_size, num_features, 1))

    @staticmethod
    def transform_peakpick_from_dict(data_dict):
        """
        peak picking from dict data
        input : float32 array
        """
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

    # tranform 2D array to dict
    @staticmethod
    def transform_arr_to_dict(arr_data):
        result_dict = {}
        for code, drum in DETECT_CODE2DRUM.items():
            result_dict[drum] = [row[code] for row in arr_data]
        return result_dict

    # tranform dict to 2D array (detect)
    @staticmethod
    def transform_dict_to_arr(dict_data):
        """
        {'OH': [0., 0., 0., ..., 0., 0., 0.], 'TT': [0., 0., 0., ..., 0., 0., 0.], 'SD': [0., 0., 0., ..., 0., 0., 0.], 'KK': [0., 0., 0., ..., 0., 0., 0.]}
        =>
        [[0,0,0,0],
        [0,0,0,0],
        ...
        [0,0,0,0]]
        """
        result_arr = np.stack([dict_data[key] for key in dict_data.keys()], axis=1)
        return result_arr

    def create_dataset(self):
        # Implement model
        pass

    def print_dataset_shape(self):
        print("x_train : ", self.x_train.shape)
        print("y_train : ", self.y_train.shape)
        print("x_val : ", self.x_val.shape)
        print("y_val : ", self.y_val.shape)
        print("x_test : ", self.x_test.shape)
        print("y_test : ", self.y_test.shape)

    def create(self):
        # Implement model creation logic
        pass

    def train(self):
        # Implement model train logic
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, mode="auto"
        )

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.training_epochs,
            callbacks=[early_stopping],
        )

        stopped_epoch = early_stopping.stopped_epoch
        print("--! finish train : stopped_epoch >> ", stopped_epoch, " !--")

        return history

    def data_2d_reshape(self, data):
        return data.reshape((-1, self.n_classes))

    def evaluate(self):
        # Implement model evaluation logic
        print("\n# Evaluate on test data")

        results = self.model.evaluate(
            self.x_test, self.y_test, batch_size=self.batch_size
        )
        print("test loss:", results[0])
        print("test accuracy:", results[1])

        try:
            # -- predict
            y_pred = self.model.predict(self.x_test)

            # -- reshape
            y_test_data = self.data_2d_reshape(self.y_test)
            y_pred = self.data_2d_reshape(y_pred)

            # -- binary
            if self.method_type == METHOD_DETECT:
                # y array -> y dict -> peakpick dict -> y array
                y_test_data = DataProcessing.convert_array_dtype_float32(y_test_data)
                y_test_data = BaseModel.transform_arr_to_dict(y_test_data)
                y_test_data = BaseModel.transform_peakpick_from_dict(y_test_data)
                y_test_data = BaseModel.transform_dict_to_arr(y_test_data)

                y_pred = BaseModel.transform_arr_to_dict(y_pred)
                y_pred = BaseModel.transform_peakpick_from_dict(y_pred)
                y_pred = BaseModel.transform_dict_to_arr(y_pred)
            else:
                y_pred = np.where(y_pred > self.predict_standard, 1.0, 0.0)

            # confusion matrix & precision & recall
            print("-- ! confusion matrix ! --")
            print(multilabel_confusion_matrix(y_test_data, y_pred))

            print("-- ! classification report ! --")
            print(classification_report(y_test_data, y_pred))
        except Exception as e:
            print(e)

    def extract_feature(self, data_path: str = f"{ROOT_PATH}/{RAW_PATH}"):
        """
        모델 학습에 사용하는 피쳐 추출하는 함수
        """
        audio_paths = DataProcessing.get_paths(data_path)
        FeatureExtractor.feature_extractor(
            audio_paths, self.method_type, self.feature_type, self.feature_extension
        )

    def run(self):
        """
        데이터셋 생성, 모델 생성, 학습, 평가, 모델 저장 파이프라인
        """
        self.create_dataset()
        self.create()
        self.train()
        self.evaluate()
        self.save()

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        pass
