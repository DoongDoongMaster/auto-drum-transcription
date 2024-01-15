import librosa
import tensorflow as tf

from typing import List

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from model.base_model import BaseModel
from constant import DETECT, STFT, MILLISECOND


class SeparateDetectModel(BaseModel):
    def __init__(
        self, training_epochs=40, opt_learning_rate=0.001, batch_size=20, unit_number=16
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=DETECT,
            feature_type=STFT,
        )
        self.unit_number = unit_number
        self.predict_standard = 0.8
        # self.n_rows = self.feature_extractor.feature_param["n_times"]
        # self.n_columns = self.feature_extractor.feature_param["n_features"]
        # self.n_classes = self.feature_extractor.feature_param["n_classes"]
        # STFT feature type
        self.n_rows = self.feature_extractor.feature_param["n_times"]
        self.n_columns = self.feature_extractor.feature_param["n_fft"] // 2 + 1
        self.n_classes = self.feature_extractor.feature_param["n_classes"]
        self.hop_length = self.feature_extractor.feature_param["hop_length"]
        self.win_length = self.feature_extractor.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        # Implement input reshaping logic
        return tf.reshape(
            data,
            [
                -1,
                self.n_rows,
                self.n_columns,
            ],
        )

    def input_label_reshape(self, data):
        return tf.reshape(data, [-1, self.n_rows * self.n_classes])

    def output_reshape(self, data):
        return tf.reshape(data, [-1, self.n_rows, self.n_classes])

    def create_dataset(self):
        super().create_dataset()

        self.y_train = self.input_label_reshape(self.y_train)
        self.y_val = self.input_label_reshape(self.y_val)
        self.y_test = self.input_label_reshape(self.y_test)

    def create(self):
        # Implement model creation logic
        self.model = Sequential()

        self.model.add(
            Bidirectional(
                SimpleRNN(
                    self.unit_number,
                    return_sequences=True,
                    input_shape=(self.n_rows, self.n_columns),
                    activation="tanh",
                )
            )
        )
        self.model.add(
            Bidirectional(
                SimpleRNN(self.unit_number, return_sequences=True, activation="tanh")
            )
        )
        self.model.add(
            Bidirectional(
                SimpleRNN(self.unit_number, return_sequences=True, activation="tanh")
            )
        )

        # Flatten layer
        self.model.add(Flatten())

        # dense layer
        self.model.add(Dense(self.n_rows * self.n_classes, activation="softmax"))

        self.model.build((None, self.n_rows, self.n_columns))
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
        return self.onset_detection.get_rhythm(
            audio_wav, bpm, onsets_arr, is_our_train_data=True
        )

    """
    -- input  : time stamp마다 onset 확률 (모델 결과)
    -- output : 
        - onsets 배열
        - 일정 확률 이상으로 예측된 악기 추출
          [몇 번째 onset, [악기]]
          ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_onsets_instrument(self, predict_data) -> List[float]:
        onsets_arr = []
        drum_instrument = []

        for i in range(len(predict_data)):
            is_onset = False  # predict standard 이상 (1) 인 j가 하나라도 있다면 onset으로 판단
            drums = []
            for j in range(self.n_classes):
                if predict_data[i][j] > self.predict_standard:
                    is_onset = True
                    drums.append(j)
            if is_onset:
                onsets_arr.append(i / self.sample_rate)
                drum_instrument.append([len(onsets_arr), drums])

        return onsets_arr, drum_instrument

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)

        # -- cut delay
        new_audio = self.data_processing.trim_audio_first_onset(
            audio, delay / MILLISECOND
        )

        # -- wav to feature
        audio_feature = self.feature_extractor.audio_to_feature(new_audio)

        # -- input reshape
        audio_feature = self.input_reshape(audio_feature)

        # -- predict
        predict_data = self.model.predict(audio_feature)

        # -- output reshape
        predict_data = self.output_reshape(predict_data)[0]

        # -- get onsets
        onsets_arr, drum_instrument = self.get_predict_onsets_instrument(predict_data)

        # -- rhythm
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        return {"instrument": drum_instrument, "rhythm": bar_rhythm}
