from pyexpat import model
import librosa
import tensorflow as tf

from typing import List

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    GRU,
    BatchNormalization,
    LSTM,
    InputLayer,
    Dropout,
    Flatten,
    Bidirectional,
    SimpleRNN,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
from data.data_labeling import DataLabeling
from sklearn.preprocessing import StandardScaler


from model.base_model import BaseModel
from data.rhythm_detection import RhythmDetection
from data.data_processing import DataProcessing
from feature.audio_to_feature import AudioToFeature
from constant import (
    METHOD_DETECT,
    MEL_SPECTROGRAM,
    MILLISECOND,
    CHUNK_LENGTH,
    SAMPLE_RATE,
)
from tensorflow.keras.layers import (
    Dense,
    GRU,
)


class SeparateDetectModel(BaseModel):
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
        # STFT feature type
        self.n_rows = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param["hop_length"]
        self.n_columns = self.feature_param["n_fft"] // 2 + 1
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        self.win_length = self.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        # Implement input reshaping logic
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        chunk_size = 100
        data = BaseModel.split_data(data, chunk_size)

        return data

    def input_label_reshape(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        chunk_size = 100
        data = BaseModel.split_data(data, chunk_size)

        return data

    def output_reshape(self, data):
        return tf.reshape(data, [-1, self.n_rows, self.n_classes])

    def create_dataset(self):
        super().create_dataset()

        # self.y_train = self.input_label_reshape(self.y_train)
        # self.y_val = self.input_label_reshape(self.y_val)
        # self.y_test = self.input_label_reshape(self.y_test)

    def create(self):
        self.model = Sequential()
        # ------------------------------------------------------------
        # self.model.add(InputLayer(input_shape=(128, 1)))
        # self.model.add(LSTM(128))

        # self.model.add(
        #     LSTM(
        #         128,
        #         return_sequences=True,
        #         input_shape=(self.x_train[1], self.x_train[2]),
        #     )
        # )
        # self.model.add(Dropout(0.25))
        # self.model.add(Bidirectional(LSTM(128)))
        # self.model.add(Dropout(0.25))

        # self.model.add(
        #     LSTM(64, input_shape=(400, 128), return_sequences=True)
        # )  # 64는 LSTM 레이어의 유닛 수로 조절 가능

        # # ------------------------------------------------------------
        # self.model.add(TimeDistributed(Dense(4, activation="sigmoid")))
        # # self.model.add(Dense(4, activation="softmax"))

        # ------------------------------------------------------------
        # -- input_shape=(timesteps, input_features)
        self.model.add(LSTM(256, input_shape=(100, 128), return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.3))
        self.model.add(TimeDistributed(Dense(4, activation="sigmoid")))
        # self.model.add(Dense(4, activation="sigmoid"))

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
        - 일정 확률 이상으로 예측된 악기 추출
          [몇 번째 onset, [악기]]
          ex. [[1, [1, 7]], [2, [1]], [3, [1]], [4, [1]], ...
    """

    def get_predict_onsets_instrument(self, predict_data) -> List[float]:
        onsets_arr = []
        drum_instrument = []

        for i in range(len(predict_data)):
            is_onset = (
                False  # predict standard 이상 (1) 인 j가 하나라도 있다면 onset으로 판단
            )
            drums = []
            for j in range(self.n_classes):
                if predict_data[i][j] > self.predict_standard:
                    is_onset = True
                    drums.append(j)
            if is_onset:
                onsets_arr.append(i * self.hop_length / SAMPLE_RATE)
                drum_instrument.append([len(onsets_arr), drums])

        return onsets_arr, drum_instrument

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio = SeparateDetectModel.load_audio(wav_path)

        # -- cut delay
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)

        # -- wav to feature
        audio_feature = AudioToFeature.extract_feature(
            new_audio, self.method_type, self.feature_type
        )

        # -- input reshape
        audio_feature = self.input_reshape(audio_feature)
        print("dㅇ러ㅏㅣㅇ리ㅓㅏㅣㅁ;ㅏ이11!<<<>>>>>>", audio_feature.shape)

        # -- predict
        predict_data = self.model.predict(audio_feature)
        print("아아아아아ㅏㅏ아아!!!!!!!!!!!!!!!>>>", predict_data.shape)

        DataLabeling.show_label_plot(predict_data)

        # # -- get onsets
        # onsets_arr, drum_instrument = self.get_predict_onsets_instrument(predict_data)

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
