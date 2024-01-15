import librosa
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

from model.base_model import BaseModel
from constant import CLASSIFY, MFCC, MILLISECOND


class SegmentClassifyModel(BaseModel):
    def __init__(self, training_epochs=40, opt_learning_rate=0.001, batch_size=20):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=CLASSIFY,
            feature_type=MFCC,
        )
        self.predict_standard = 0.5
        self.n_row = self.feature_extractor.feature_param["n_features"]
        self.n_columns = self.feature_extractor.feature_param["n_times"]
        self.n_channels = self.feature_extractor.feature_param["n_channels"]
        self.n_classes = self.feature_extractor.feature_param["n_classes"]
        self.load_model()

    def input_reshape(self, data):
        # Implement input reshaping logic
        return tf.reshape(
            data,
            [
                -1,
                self.n_row,
                self.n_columns,
                self.n_channels,
            ],
        )

    def create(self):
        # Implement model creation logic
        self.model = Sequential()

        self.model.add(
            layers.Conv2D(
                input_shape=(self.n_row, self.n_columns, self.n_channels),
                filters=16,
                kernel_size=(4, 4),
                activation="relu",
                padding="same",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.2))

        self.model.add(
            layers.Conv2D(
                filters=16 * 2, kernel_size=(4, 4), activation="relu", padding="same"
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.2))

        self.model.add(
            layers.Conv2D(
                filters=16 * 3, kernel_size=(4, 4), activation="relu", padding="same"
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(units=self.n_classes, activation="sigmoid"))

        self.model.summary()

        opt = Adam(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    """
    -- 전체 wav 주어졌을 때, 한 마디에 대한 rhythm 계산
    """

    def get_bar_rhythm(self, audio_wav, bpm):
        return self.onset_detection.get_rhythm(audio_wav, bpm, is_our_train_data=True)

    """
    -- input  : onset마다 예측한 악기 확률
    -- output : 일정 확률 이상으로 예측된 악기 추출
                [몇 번째 onset, [악기]]
                ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_result(self, predict):
        # 각 행에서 threshold를 넘는 값의 인덱스 찾기
        indices_above_threshold = np.argwhere(predict > self.predict_standard)

        current_row = indices_above_threshold[0, 0]
        result = []
        cols = []
        for index in indices_above_threshold:
            row, col = index
            if row != current_row:
                tmp = [row, cols]
                result.append(tmp)
                current_row = row
                cols = []
            cols.append(col)
        return result

    """
    -- input  : 1 wav
    -- output : 각 onset에 대한 악기 종류 분류
    """

    def get_drum_instrument(self, audio):
        # -- trimmed audio
        trimmed_audios = self.data_processing.trim_audio_per_onset(audio)

        # -- trimmed feature
        predict_data = []
        for _, taudio in enumerate(trimmed_audios):
            trimmed_feature = self.feature_extractor.audio_to_feature(taudio)
            predict_data.append(trimmed_feature)

        # -- reshape
        predict_data = self.input_reshape(predict_data)

        # -- predict
        predict_data = self.model.predict(predict_data)

        return self.get_predict_result(predict_data)

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        # -- instrument
        drum_instrument = self.get_drum_instrument(audio)
        # -- rhythm
        new_audio = self.data_processing.trim_audio_first_onset(
            audio, delay / MILLISECOND
        )
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm)

        return {"instrument": drum_instrument, "rhythm": bar_rhythm}
