import os
import numpy as np
import tensorflow as tf

from glob import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect
from feature.feature_extractor import FeatureExtractor
from constant import (
    SAMPLE_RATE,
    ROOT_PATH,
    NEW_PATH,
    RAW_PATH,
    PROCESSED_FEATURE,
)


class BaseModel:
    def __init__(
        self, training_epochs, opt_learning_rate, batch_size, method_type, feature_type
    ):
        self.model = None
        self.training_epochs = training_epochs
        self.opt_learning_rate = opt_learning_rate
        self.batch_size = batch_size
        self.method_type = method_type
        self.feature_type = feature_type
        self.sample_rate = SAMPLE_RATE
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.onset_detection = OnsetDetect(self.sample_rate)
        self.data_processing = DataProcessing(ROOT_PATH)
        self.feature_extractor = FeatureExtractor(
            data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
            method_type=method_type,
            feature_type=feature_type,
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
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        print("-- ! load model: ", model_files[0])
        self.model = tf.keras.models.load_model(model_files[0])

    def load_data(self):
        # # feature file 존재안한다면 -> raw data feature 생성해서 저장하기
        # if os.path.exists(self.feature_extractor.save_path) is False:
        #     print("-- ! 기존 raw data에서 feature file 새로 생성 ! --")
        #     paths = self.data_processing.get_paths(self.data_processing.raw_data_path)
        #     self.feature_extractor.feature_extractor(paths)

        if DataProcessing.is_exist_data_in_folder(
            f"{ROOT_PATH}/{NEW_PATH}"
        ):  # 새로운 데이터 있는지 확인
            print("-- ! 새로운 데이터 존재 ! --")
            feature_file_list = self.feature_extractor.load_feature_file_all()
            for feature_file in feature_file_list:
                feature_extension = os.path.splitext(feature_file)[1][1:]  # 파일 확장자
                feature_type_new = os.path.basename(feature_file)[:-4]
                method_type_new = feature_file.split("/")[-2]  # 뒤에서 2번째 인덱스
                print(
                    f"-- ! 기존 feature update: {method_type_new}/{feature_type_new}.{feature_extension}"
                )
                feature_extractor_new = FeatureExtractor(
                    data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
                    method_type=method_type_new,
                    feature_type=feature_type_new,
                    feature_extension=feature_extension,
                )
                new_data_paths = self.data_processing.get_paths(
                    self.data_processing.new_data_path
                )
                feature_extractor_new.feature_extractor(new_data_paths)

            self.data_processing.move_new_to_raw(
                f"{ROOT_PATH}/{NEW_PATH}", f"{ROOT_PATH}/{RAW_PATH}"
            )

        # feature 불러오기
        return self.feature_extractor.load_feature_file()

    def input_reshape(self, data) -> np.ndarray:
        # Implement input reshaping logic
        pass

    """
    -- load data from data file
    -- Implement dataset split feature & label logic
    """

    def create_dataset(self):
        # Implement dataset split feature & label logic
        feature_df = self.load_data()

        # X = np.array(feature_df.feature.tolist())

        # mel-spec 1~128
        X = feature_df.drop(["label"], axis=1).to_numpy()
        y = feature_df["label"].to_numpy()
        # y = np.array(feature_df.label.tolist())

        # -- split train, val, test
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        del X
        del y

        x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
            x_train_temp, y_train_temp, test_size=0.2, random_state=42
        )

        del x_train_temp
        del y_train_temp

        # input shape 조정
        # self.x_train = self.input_reshape(x_train_final)
        # self.x_val = self.input_reshape(x_val_final)
        # self.x_test = self.input_reshape(x_test)

        scaler = StandardScaler()
        x_train_final = scaler.fit_transform(x_train_final)
        x_val_final = scaler.transform(x_val_final)
        x_test = scaler.transform(x_test)

        # Reshape for model input
        x_train_final = np.expand_dims(x_train_final, axis=-1)
        x_val_final = np.expand_dims(x_val_final, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        self.x_train = x_train_final
        self.x_val = x_val_final
        self.x_test = x_test
        self.y_train = y_train_final
        self.y_val = y_val_final
        self.y_test = y_test

        # -- print shape
        self.print_dataset_shape()

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

    def evaluate(self):
        # Implement model evaluation logic
        print("\n# Evaluate on test data")

        results = self.model.evaluate(
            self.x_test, self.y_test, batch_size=self.batch_size
        )
        print("test loss:", results[0])
        print("test accuracy:", results[1])

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        pass
