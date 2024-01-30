import librosa
import numpy as np
import tensorflow as tf
import numpy as np

from typing import List
from sklearn.preprocessing import StandardScaler
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
from model.base_model import BaseModel
from constant import (
    METHOD_RHYTHM,
    STFT,
    MILLISECOND,
    CHUNK_LENGTH,
    SAMPLE_RATE,
    MEL_SPECTROGRAM,
)

from keras.utils import to_categorical
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
        super().create_dataset()

        # self.x_train = self.input_reshape(self.x_train)
        # self.x_val = self.input_reshape(self.x_val)
        # self.x_test = self.input_reshape(self.x_test)

        # print(">>>>>>>>>>>>>>>.self.x_train.shape", self.x_train)
        # self.y_train = self.input_label_reshape(self.y_train)
        # self.y_val = self.input_label_reshape(self.y_val)
        # self.y_test = self.input_label_reshape(self.y_test)

        # self.y_train = to_categorical(self.y_train, num_classes=self.n_classes * 1200)
        # self.y_val = to_categorical(self.y_val, num_classes=self.n_classes * 1200)
        # self.y_test = to_categorical(self.y_test, num_classes=self.n_classes * 1200)

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

        # self.model.add(
        #     Bidirectional(
        #         SimpleRNN(
        #             self.unit_number,
        #             return_sequences=True,
        #             input_shape=(self.n_rows, self.n_columns),
        #             activation="tanh",
        #         )
        #     )
        # )
        # self.model.add(
        #     Bidirectional(
        #         SimpleRNN(self.unit_number, return_sequences=True, activation="tanh")
        #     )
        # )
        # self.model.add(
        #     Bidirectional(
        #         SimpleRNN(self.unit_number, return_sequences=True, activation="tanh")
        #     )
        # )

        # # Flatten layer
        # self.model.add(Flatten())

        # # dense layer
        # self.model.add(Dense(self.n_rows * self.n_classes, activation="softmax"))

        # self.model.build((None, self.n_rows, self.n_columns))
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

    def get_predict_onsets_instrument(self, predict_data) -> List[float]:
        onsets_arr = []

        for i in range(len(predict_data)):
            if predict_data[i] > self.predict_standard:
                onsets_arr.append(i / SAMPLE_RATE)

        return onsets_arr

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        # -- cut delay
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)

        # -- wav to feature
        audio_feature = AudioToFeature.extract_feature(
            new_audio, self.method_type, self.feature_type
        )

        # # -- input reshape
        # audio_feature = self.input_reshape(audio_feature)

        # ======================== new work ==============================
        # scaler = StandardScaler()
        # audio_feature = scaler.fit_transform(audio_feature)
        # Reshape for model input
        audio_feature = np.expand_dims(audio_feature, axis=-1)

        # -- predict
        predict_data = self.model.predict(audio_feature)

        # print(predict_data)

        # -- output reshape
        predict_data = self.output_reshape(predict_data)[0]

        DataLabeling.show_label_plot(predict_data)

        # -- get onsets
        onsets_arr = self.get_predict_onsets_instrument(predict_data)

        # -- rhythm
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        return {"rhythm": bar_rhythm}
