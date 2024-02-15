import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from glob import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
)

from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from constant import (
    SAMPLE_RATE,
    PKL,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    FEATURE_PARAM,
    ROOT_PATH,
    RAW_PATH,
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

    def load_model(self):
        """
        -- method_type과 feature type에 맞는 가장 최근 모델 불러오기
        """
        model_files = glob(f"{self.save_path}_*.{self.model_save_type}")
        if model_files is None or len(model_files) == 0:
            print("-- ! No pre-trained model ! --")
            return

        model_files.sort(reverse=True)  # 최신 순으로 정렬
        print("-- ! load model: ", model_files[0])
        self.model = tf.keras.models.load_model(model_files[0])

    def input_reshape(self, data) -> np.ndarray:
        # Implement input reshaping logic
        pass

    @staticmethod
    def _get_x_y(method_type: str, feature_df: pd.DataFrame):
        if method_type == METHOD_CLASSIFY:
            X = np.array(feature_df.feature.tolist())
            y = feature_df[["HH", "ST", "SD", "KK"]].to_numpy()
            return X, y
        if method_type in METHOD_DETECT:
            # label(HH, ST, SD, KK onset 여부) | mel-1, mel-2, mel-3, ...
            X = feature_df.drop(["HH", "ST", "SD", "KK"], axis=1).to_numpy()
            y = feature_df[["HH", "ST", "SD", "KK"]].to_numpy()
            return X, y
        if method_type in METHOD_RHYTHM:
            # label(onset 여부) | mel-1, mel-2, mel-3, ...
            X = feature_df.drop(["label"], axis=1).to_numpy()
            y = feature_df["label"].to_numpy()
            return X, y

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

        # -- split train, val, test
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        self.x_train = self.input_reshape(x_train_final)
        self.x_val = self.input_reshape(x_val_final)
        self.x_test = self.input_reshape(x_test)
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

        # -- predict
        y_pred = self.model.predict(self.x_test)
        y_pred = np.where(y_pred > self.predict_standard, 1.0, 0.0)

        # confusion matrix & precision & recall
        print("-- ! confusion matrix ! --")
        print(multilabel_confusion_matrix(self.y_test, y_pred))

        print("-- ! classification report ! --")
        print(classification_report(self.y_test, y_pred))

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

    @staticmethod
    def load_audio(path):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        audio = librosa.effects.percussive(audio)
        return audio
