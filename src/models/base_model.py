import os
import numpy as np
import tensorflow as tf

from glob import glob
from datetime import datetime

from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect
from feature.feature_extractor import FeatureExtractor
from constant import (
    ONSET_DURATION,
    SAMPLE_RATE,
    ROOT_PATH,
    PROCESSED_FEATURE,
    FEATURE_PARAM,
)


class BaseModel:
    def __init__(
        self, training_epochs, opt_learning_rate, batch_size, method_type, feature_type
    ):
        self.data = None
        self.model = None
        self.training_epochs = training_epochs
        self.opt_learning_rate = opt_learning_rate
        self.batch_size = batch_size
        self.method_type = method_type
        self.feature_type = feature_type
        self.sample_rate = SAMPLE_RATE
        self.onset_duration = ONSET_DURATION
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.onset_detection = OnsetDetect(self.sample_rate, self.onset_duration)
        self.data_processing = DataProcessing(ROOT_PATH)
        self.feature_extractor = FeatureExtractor(
            data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
            method_type=method_type,
            feature_type=feature_type,
            feature_param=FEATURE_PARAM[method_type][feature_type],
        )
        self.save_path = f"../models/{method_type}_{feature_type}"
        self.model_save_type = "h5"

    def save(self):
        # 현재 날짜와 시간 가져오기
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 모델 저장
        model_path = f"{self.save_path}_{date_time}.{self.model_save_type}"
        self.model.save(model_path)
        print("--! save model: ", model_path)

    def load_model(self):
        model_files = glob(f"{self.save_path}_*.{self.model_save_type}")
        if model_files is None:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        print("-- ! load model: ", model_files[0])
        self.model = tf.keras.models.load_model(model_files[0])

    def load_data(self):
        # feature file 존재안한다면 -> raw data feature 생성해서 저장하기
        if self.feature_extractor.load_feature_csv() is None:
            paths = self.data_processing.get_paths(self.data_processing.raw_data_path)
            self.feature_extractor.feature_extractor(paths)

        if self.data_processing.is_exist_new_data():  # 새로운 데이터 있는지 확인
            feature_csv_list = self.feature_extractor.load_feature_csv_all()
            for feature_csv in feature_csv_list:
                feature_type_new = os.path.basename(feature_csv)[:-4]
                method_type_new = feature_csv.split("/")[-2:][0]
                feature_extractor_new = FeatureExtractor(
                    data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
                    method_type=method_type_new,
                    feature_type=feature_type_new,
                    feature_param=FEATURE_PARAM[method_type_new][feature_type_new],
                )
                new_data_paths = self.data_processing.get_paths(
                    self.data_processing.new_data_path
                )
                feature_extractor_new.feature_extractor(new_data_paths)

            self.data_processing.move_new_to_raw()

        # feature 불러오기
        self.data = self.feature_extractor.load_feature_csv()

    def input_reshape(self, data) -> np.ndarray:
        # Implement input reshaping logic
        pass

    def create_dataset(self):
        # Implement dataset split feature & label logic
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
        pass

    def evaluate(self):
        # Implement model evaluation logic
        pass

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        pass
