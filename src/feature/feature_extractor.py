import os
import librosa
import numpy as np
import pandas as pd

from ast import literal_eval
from typing import List
from glob import glob

from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect

from constant import (
    PATTERN_DIR,
    PER_DRUM_DIR,
    SAMPLE_RATE,
    ONSET_DURATION,
    PATTERN2CODE,
    ONEHOT_DRUM2CODE,
    MFCC,
    STFT,
    CODE2DRUM,
    CLASSIFY,
    DETECT,
    ROOT_PATH,
)

"""
데이터에서 feature를 추출하고, 라벨링하고, 저장하는 클래스
"""


class FeatureExtractor:
    def __init__(
        self,
        data_root_path,
        method_type,
        feature_type,
        feature_param: dict,
    ):
        self.data_root_path = data_root_path
        self.method_type = method_type
        self.feature_type = feature_type
        self.sample_rate = SAMPLE_RATE
        self.feature_param = feature_param
        self.save_path = f"{data_root_path}/{method_type}/{feature_type}.csv"
        self.onset_detection = OnsetDetect(SAMPLE_RATE, ONSET_DURATION)
        self.data_processing = DataProcessing(data_root_path=ROOT_PATH)

    """
    -- feature 추출한 파일 불러오기
    """

    def load_feature_csv(self):
        data_feature_label = None
        if os.path.exists(self.save_path):  # 추출된 feature 존재 한다면
            print("-- ! 기존 feature loading ! --")
            data_feature_label = pd.read_csv(
                self.save_path,
                index_col=0,
                converters={"feature": literal_eval, "label": literal_eval},
            )
            print(
                "-- ! 로딩 완료 ! --",
                "data shape:",
                data_feature_label.shape,
            )
            print("-- ! features ! -- ")
            print(data_feature_label)

        return data_feature_label

    """
    -- feature csv file 모두 불러오기
    """

    def load_feature_csv_all(self):
        feature_csv_list = glob(f"{self.data_root_path}/**/*.csv", recursive=True)
        return feature_csv_list

    """
    -- feature 파일 저장하기
    """

    def save_feature_csv(self, features: pd.DataFrame):
        # Save csv file
        features.to_csv(self.save_path, sep=",")

        print("-- ! 완료 & 새로 저장 ! --")
        print("-- ! location: ", self.save_path)
        print("-- ! features shape:", features.shape)

    """
    -- mfcc feature 추출
    """

    def audio_to_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.feature_param["n_features"]
        )
        pad_width = self.feature_param["n_times"] - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
        print(
            "-- length:",
            audio.shape[0] / float(self.sample_rate),
            "secs, ",
            "mfccs:",
            mfccs.shape,
        )
        return mfccs

    """
    -- stft feature 추출
    """

    def audio_to_stft(self, audio: np.ndarray) -> np.ndarray:
        # translate STFT
        stft = librosa.stft(
            y=audio,
            n_fft=self.feature_param["n_fft"],
            hop_length=self.feature_param["hop_length"],
            win_length=self.feature_param["win_length"],
            window="hann",
        )
        stft = np.abs(stft, dtype=np.float64)
        if stft.shape[1] < self.feature_param["n_times"]:
            stft_new = np.pad(
                stft,
                pad_width=((0, 0), (0, self.feature_param["n_times"] - stft.shape[1])),
                mode="constant",
            )
        else:
            stft_new = stft[:, : self.feature_param["n_times"]]
        stft_new = np.transpose(stft_new)  # row: time, col: feature

        print(
            "-- length:",
            audio.shape[0] / float(self.sample_rate),
            "secs, ",
            "stft:",
            stft_new.shape,
        )
        return stft_new

    """
    -- feature type에 따라 feature 추출
    """

    def audio_to_feature(self, audio: np.ndarray) -> np.ndarray:
        if self.feature_type == MFCC:
            return self.audio_to_mfcc(audio)
        elif self.feature_type == STFT:
            return self.audio_to_stft(audio)

    """
    -- 우리 데이터 기준 classify type (trimmed data) 라벨링
    """

    def get_label_classify_data(self, idx: int, path: str) -> List[int]:
        file_name = os.path.basename(path)  # extract file name
        if PATTERN_DIR in path:  # -- pattern
            pattern_name = file_name[:2]  # -- P1
            label = PATTERN2CODE[pattern_name][idx]
        elif PER_DRUM_DIR in path:  # -- per drum
            drum_name = file_name[:2]  # -- CC
            label = ONEHOT_DRUM2CODE[drum_name]
        return label

    """
    -- 우리 데이터 기준 detect type (sequence data) 라벨링 
        onset position : 1
        onset position with ONSET_DURATION : 0.5
        extra : 0
    """

    def get_audio_position(self, time) -> int:
        return (int)(time * self.sample_rate / self.feature_param["hop_length"])

    def get_label_detect_data(self, path: str) -> List[List[int]]:
        file_name = os.path.basename(path)
        audio, _ = librosa.load(path, sr=self.sample_rate)
        onsets_arr = self.onset_detection.onset_detection(audio)

        labels = [[0.0] * len(CODE2DRUM) for _ in range(self.feature_param["n_times"])]
        pattern_idx = 0
        for onset in onsets_arr:
            soft_start_position = self.get_audio_position(
                max((onset - ONSET_DURATION), 0)
            )
            onset_position = self.get_audio_position(onset)
            soft_end_position = self.get_audio_position(onset + ONSET_DURATION)

            if any(drum in file_name for _, drum in CODE2DRUM.items()):  # per drum
                one_hot_label: List[int] = ONEHOT_DRUM2CODE[file_name[:2]]
            else:  # pattern
                pattern_name = file_name[:2]  # -- P1
                one_hot_label: List[int] = PATTERN2CODE[pattern_name][pattern_idx]
                pattern_idx += 1
            for i in range(soft_start_position, soft_end_position):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = np.array(one_hot_label) / 2
            labels[(int)(onset_position)] = one_hot_label

        return labels

    """
    -- classify type feature, label 추출
    """

    def classify_feature_extractor(self, audio_paths: List[str]) -> pd.DataFrame:
        data_feature_label = []

        print(f"-- ! ADT type : {self.method_type} ! --")
        print(f"-- ! {self.feature_type} feature extracting ! --")
        for path in audio_paths:
            print("-- curret file: ", path)
            # -- librosa feature load
            audio, _ = librosa.load(path, sr=self.sample_rate, res_type="kaiser_fast")
            # -- trimmed audio
            trimmed_audios = self.data_processing.trim_audio_per_onset(audio)
            # -- trimmed feature
            for idx, taudio in enumerate(trimmed_audios):
                trimmed_feature = self.audio_to_feature(taudio)
                # -- label: 드럼 종류
                label = self.get_label_classify_data(idx, path)
                data_feature_label.append([trimmed_feature.tolist(), label])

        feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        return feature_df

    """
    -- detect type feature, label 추출
    """

    def detect_feature_extractor(self, audio_paths: List[str]) -> pd.DataFrame:
        data_feature_label = []

        print(f"-- ! ADT type : {self.method_type} ! --")
        print(f"-- ! {self.feature_type} feature extracting ! --")
        for path in audio_paths:
            print("-- curret file: ", path)
            # -- librosa feature load
            audio, _ = librosa.load(path, sr=self.sample_rate, res_type="kaiser_fast")
            feature = self.audio_to_feature(audio)
            # -- label: 드럼 종류마다 onset 여부
            label = self.get_label_detect_data(path)
            data_feature_label.append([feature.tolist(), label])

        feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        return feature_df

    """ 
    -- method type에 따라 feature, label 추출 후 저장
    """

    def feature_extractor(self, audio_paths):
        features_df_origin = self.load_feature_csv()  # load feature csv file

        features_df_new = None
        if self.method_type == CLASSIFY:
            features_df_new = self.classify_feature_extractor(audio_paths)
        elif self.method_type == DETECT:
            features_df_new = self.detect_feature_extractor(audio_paths)

        # Convert into a Panda dataframe & Add dataframe
        features_total_df = features_df_new
        if features_df_origin is not None:
            features_total_df = pd.concat([features_df_origin, features_df_new])

        # Save csv file
        self.save_feature_csv(features_total_df)
