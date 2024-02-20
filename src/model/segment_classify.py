import librosa
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from model.base_model import BaseModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from data.rhythm_detection import RhythmDetection
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from constant import (
    METHOD_CLASSIFY,
    MFCC,
    MILLISECOND,
    SAMPLE_RATE,
    CLASSIFY_DURATION,
    PKL,
    CODE2DRUM,
)


class SegmentClassifyModel(BaseModel):
    def __init__(
        self,
        training_epochs=40,
        opt_learning_rate=0.001,
        batch_size=20,
        feature_type=MFCC,
        feature_extension=PKL,
    ):
        super().__init__(
            training_epochs=training_epochs,
            opt_learning_rate=opt_learning_rate,
            batch_size=batch_size,
            method_type=METHOD_CLASSIFY,
            feature_type=feature_type,
            feature_extension=feature_extension,
        )
        self.predict_standard = 0.5
        self.n_row = self.feature_param["n_mfcc"]
        self.n_columns = (
            int(CLASSIFY_DURATION * SAMPLE_RATE) // self.feature_param["hop_length"]
        )
        self.n_channels = self.feature_param["n_channels"]
        self.n_classes = self.feature_param["n_classes"]
        self.hop_length = self.feature_param["hop_length"]
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

    def create_dataset(self):
        """
        -- load data from data file
        -- Implement dataset split feature & label logic
        """
        super().create_dataset()

    def create(self):
        # Implement model creation logic
        self.model = Sequential()

        self.model.add(
            layers.Conv2D(
                input_shape=(self.n_row, self.n_columns, self.n_channels),
                filters=64,
                kernel_size=(3, 3),
                activation="tanh",
                padding="same",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(layers.Dropout(0.2))

        self.model.add(
            layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="tanh", padding="same"
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(layers.Dropout(0.2))

        self.model.add(
            layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="tanh", padding="same"
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(layers.Dropout(0.2))

        self.model.add(layers.GlobalAveragePooling2D())

        self.model.add(layers.Reshape((64, 1), input_shape=(None, 64, 1)))
        self.model.add(
            layers.Bidirectional(
                layers.LSTM(8, input_shape=(None, 64, 1), return_sequences=True)
            )
        )
        self.model.add(layers.Bidirectional(layers.LSTM(10)))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=self.n_classes, activation="sigmoid"))

        self.model.summary()

        opt = RMSprop(learning_rate=self.opt_learning_rate)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    """
    -- 전체 wav 주어졌을 때, 한 마디에 대한 rhythm 계산
    """

    def get_bar_rhythm(self, audio_wav, bpm, onsets_arr):
        return RhythmDetection.get_rhythm(audio_wav, bpm, onsets_arr)

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
                tmp = [current_row, cols]
                result.append(tmp)
                current_row = row
                cols = []
            cols.append(col)
        result.append([current_row, cols])
        return result

    """
    -- input  : 1 wav
    -- output : 각 onset에 대한 악기 종류 분류
    """

    def get_drum_instrument(self, audio):
        # -- trimmed audio
        onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)
        trimmed_audios = DataProcessing.trim_audio_per_onset(audio, onsets_arr)

        # -- trimmed feature
        predict_data = []
        for _, taudio in enumerate(trimmed_audios):
            trimmed_feature = AudioToFeature.extract_feature(
                taudio, self.method_type, self.feature_type
            )
            predict_data.append(trimmed_feature)

        # -- reshape
        predict_data = self.input_reshape(predict_data)

        # -- predict
        predict_data = self.model.predict(predict_data)

        print("-- ! classify 방법 예측 결과 ! --")
        np.set_printoptions(precision=2, suppress=True)
        print(predict_data)

        return self.get_predict_result(predict_data)

    def predict(self, wav_path, bpm, delay):
        # Implement model predict logic
        audio = FeatureExtractor.load_audio(wav_path)

        # -- instrument
        drum_instrument = self.get_drum_instrument(audio)
        # -- rhythm
        onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)

        # -- 원래 정답 라벨
        true_label = DataLabeling.data_labeling(
            audio, wav_path, METHOD_CLASSIFY, hop_length=self.hop_length
        )
        # DataLabeling.show_label_dict_plot(true_label)

        # -- transport frame
        onset_dict = {v: [] for _, v in CODE2DRUM.items()}
        for data in drum_instrument:
            idx = data[0]
            instrument = data[1]
            for inst in instrument:
                onset_dict[CODE2DRUM[inst]].append(onsets_arr[idx])
        frame_length = len(audio) // self.hop_length
        frame_onset = DataLabeling._get_label_detect(
            onset_dict, frame_length, self.hop_length
        )
        DataLabeling.show_label_dict_compare_plot(true_label, frame_onset)

        # delay 제거
        new_audio = DataProcessing.trim_audio_first_onset(audio, delay / MILLISECOND)
        bar_rhythm = self.get_bar_rhythm(new_audio, bpm, onsets_arr)

        return {"instrument": drum_instrument, "rhythm": bar_rhythm}
