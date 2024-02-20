import librosa
import numpy as np
import tensorflow as tf
import numpy as np

from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    SimpleRNN,
    Flatten,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GRU,
    Reshape,
)
from tensorflow.keras.optimizers import Adam

from data.rhythm_detection import RhythmDetection
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from model.base_model import BaseModel
from constant import (
    METHOD_RHYTHM,
    STFT,
    MILLISECOND,
    CHUNK_LENGTH,
    SAMPLE_RATE,
    MEL_SPECTROGRAM,
)

from sklearn.preprocessing import StandardScaler


class RhythmDetectModel(BaseModel):
    def __init__(
        self, training_epochs=40, opt_learning_rate=0.001, batch_size=20, unit_number=16
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_RHYTHM,
            feature_type=MEL_SPECTROGRAM,
        )
        self.unit_number = unit_number
        self.predict_standard = 0.8

        # STFT feature type
        self.n_rows = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param["hop_length"]
        self.n_columns = self.feature_param["n_fft"] // 2 + 1
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        self.win_length = self.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Reshape for model input
        return np.expand_dims(data, axis=-1)

    def input_label_reshape(self, data):
        return tf.reshape(data, [-1, self.n_classes])

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

    def create(self):
        # Implement model creation logic
        self.model = Sequential()

        # Convolutional layers
        self.model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(128, 1, 1),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 1)))

        self.model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 1)))

        # Recurrent layers (BiGRU)
        self.model.add(Reshape((-1, 32)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))

        # Fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(1, activation="sigmoid"))

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
    """

    def get_predict_onsets_instrument(self, predict_data):
        peaks = librosa.util.peak_pick(
            predict_data,
            pre_max=20,
            post_max=20,
            pre_avg=10,
            post_avg=10,
            delta=0.1,
            wait=10,
        )
        return peaks

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio = FeatureExtractor.load_audio(wav_path)

        # -- cut delay
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)

        # -- wav to feature
        audio_feature = AudioToFeature.extract_feature(
            new_audio, self.method_type, self.feature_type
        )

        audio_feature = self.input_reshape(audio_feature)

        # -- predict
        predict_data = self.model.predict(audio_feature)

        # 이차원 배열을 1차원 배열로 변환
        predict_data = np.array([item[0] for item in predict_data])

        # -- (임시) test 5s ~ 15s
        chunk_samples_start = 500
        chunk_samples_end = 1500
        predict_data = predict_data[chunk_samples_start : chunk_samples_end + 1]

        # -- get onsets
        onsets_arr = self.get_predict_onsets_instrument(predict_data)

        DataLabeling.show_label_onset_plot(predict_data, onsets_arr)

        result = []
        for onset in onsets_arr:
            result.append(onset * self.hop_length / self.sample_rate)

        # -- rhythm
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm, result)

        return {"rhythm": bar_rhythm}
