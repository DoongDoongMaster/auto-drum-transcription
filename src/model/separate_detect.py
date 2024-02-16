from pyexpat import model
import librosa
import tensorflow as tf
import numpy as np

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
    GlobalAveragePooling1D,
    Conv2D,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Reshape,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from data.data_labeling import DataLabeling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from feature.feature_extractor import FeatureExtractor


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
        self.n_rows = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param["hop_length"]
        self.n_columns = self.feature_param["n_fft"] // 2 + 1
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
        self.win_length = self.feature_param["win_length"]
        self.load_model()

    def input_reshape(self, data):
        return data
        # Implement input reshaping logic
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        # data = BaseModel.split_data(data, chunk_size)

    def input_label_reshape(self, data):
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

        # --------------------------------------------------
        self.model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="tanh",
                input_shape=(60, 128, 1),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding="same", activation="tanh"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(1, 3)))

        self.model.add(Conv2D(32, (3, 3), padding="same", activation="tanh"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding="same", activation="tanh"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(1, 3)))

        # # Recurrent layers (BiGRU)
        self.model.add(Reshape((-1, 448)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))
        self.model.add(Bidirectional(GRU(50, return_sequences=True)))

        self.model.add(TimeDistributed(Dense(4, activation="sigmoid")))

        # --------------------------------------------------
        # # self.model.add(
        # #     Conv1D(32, 3, padding="same", activation="relu", input_shape=(128, 1))
        # # )
        # self.model.add(
        #     LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(60, 128))
        # )
        # self.model.add(Dense(4, activation="sigmoid"))
        # --------------------------------------------------

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
        audio = FeatureExtractor.load_audio(wav_path)

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

        # DataLabeling.show_label_plot(predict_data)

        # --------------------------------------------------------
        #         # -- instrument
        # drum_instrument = self.get_drum_instrument(audio)
        # # -- rhythm
        # onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)
        predict_data = predict_data[:2000]

        # -- 원래 정답 라벨
        true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_DETECT, hop_length=self.hop_length
        )

        y_pred_cut = {key: value[:2000] for key, value in true_label.items()}

        label_dict = {
            "HH": [row[0] for row in predict_data],
            "ST": [row[1] for row in predict_data],
            "SD": [row[2] for row in predict_data],
            "KK": [row[3] for row in predict_data],
        }

        # # -- transport frame
        # onset_dict = {v: [] for _, v in CODE2DRUM.items()}
        # for data in drum_instrument:
        #     idx = data[0]
        #     instrument = data[1]
        #     for inst in instrument:
        #         onset_dict[CODE2DRUM[inst]].append(onsets_arr[idx])
        # frame_length = len(audio) // self.hop_length
        # frame_onset = DataLabeling._get_label_detect(
        #     onset_dict, frame_length, self.hop_length
        # )
        DataLabeling.show_label_dict_compare_plot(y_pred_cut, label_dict)

        # # -- get onsets
        # onsets_arr, drum_instrument = self.get_predict_onsets_instrument(predict_data)

        # # -- rhythm
        # bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        # return {"instrument": drum_instrument, "rhythm": bar_rhythm}
        # return NULL
