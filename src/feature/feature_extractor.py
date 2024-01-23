import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pretty_midi

from ast import literal_eval
from typing import List
from glob import glob
from datetime import datetime

from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect
from feature.audio_to_feature import AudioToFeature


from constant import (
    PATTERN_DIR,
    PER_DRUM_DIR,
    SAMPLE_RATE,
    ONSET_OFFSET,
    PATTERN2CODE,
    ONEHOT_DRUM2CODE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    CODE2DRUM,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    ROOT_PATH,
    CSV,
    PKL,
    DDM_OWN,
    IDMT,
    ENST,
    E_GMD,
    FEATURE_PARAM,
    CHUNK_LENGTH,
    IMAGE_PATH,
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
        feature_extension=PKL,
    ):
        self.data_root_path = data_root_path
        self.method_type = method_type
        self.feature_type = feature_type
        self.feature_param = FEATURE_PARAM[method_type][feature_type]
        self.frame_length = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param[
            "hop_length"
        ]
        self.feature_extension = feature_extension
        self.save_path = (
            f"{data_root_path}/{method_type}/{feature_type}.{feature_extension}"
        )
        self.data_processing = DataProcessing()

    """
    -- feature 추출한 파일 불러오기
    """

    def load_feature_file(self):
        data_feature_label = None
        if os.path.exists(self.save_path):  # 추출된 feature 존재 한다면
            print("-- ! 기존 feature loading : ", self.save_path)

            if self.feature_extension == CSV:
                data_feature_label = pd.read_csv(
                    self.save_path,
                    index_col=0,
                    converters={"feature": literal_eval, "label": literal_eval},
                )
            elif self.feature_extension == PKL:
                data_feature_label = pd.read_pickle(self.save_path)

            print(
                "-- ! 로딩 완료 ! --",
                "data shape:",
                data_feature_label.shape,
            )
            print("-- ! features ! -- ")
            print(data_feature_label)

        return data_feature_label

    """
    -- feature file 모두 불러오기
    """

    def load_feature_file_all(self):
        feature_file_list = glob(f"{self.data_root_path}/**/*.*", recursive=True)
        print("-- ! feature file all load: ", feature_file_list)
        return feature_file_list

    """
    -- feature 파일 저장하기
    """

    def save_feature_file(self, features: pd.DataFrame):
        if self.feature_extension == CSV:
            # Save csv file
            features.to_csv(self.save_path, sep=",")
        elif self.feature_extension == PKL:
            # Save pickle file
            features.to_pickle(self.save_path)

        print("-- ! 완료 & 새로 저장 ! --")
        print("-- ! location: ", self.save_path)
        print("-- ! features shape:", features.shape)

    """
    -- onset을 chunk에 맞게 split
    ex. {0: [0~11], 1: [12, 23], 2: [], 3: [38], ... onset 을 12배수에 따라 split
    -> 0~11
    """

    def split_onset_match_chunk(self, onsets_arr: List[float]):
        chunk_onsets_arr = {}
        tmp = []
        current_chunk_idx = 0
        for onset_time in onsets_arr:
            if (
                current_chunk_idx * CHUNK_LENGTH <= onset_time
                and onset_time < (current_chunk_idx + 1) * CHUNK_LENGTH
            ):
                tmp.append(
                    onset_time - (current_chunk_idx * CHUNK_LENGTH)
                    if onset_time >= CHUNK_LENGTH
                    else onset_time
                )
                continue
            chunk_onsets_arr[current_chunk_idx] = tmp
            current_chunk_idx += 1
            tmp = []

        chunk_onsets_arr[current_chunk_idx] = tmp
        return chunk_onsets_arr

    """
    -- classify type feature, label 추출
    """

    def classify_feature_extractor(self, audio_paths: List[str]) -> pd.DataFrame:
        data_feature_label = []

        print(f"-- ! ADT type : {self.method_type} ! --")
        print(f"-- ! {self.feature_type} feature extracting ! --")
        for path in audio_paths:
            print("-- current file: ", path)
            # -- librosa feature load
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")

            if DDM_OWN in path:  # 우리 데이터라면
                # -- trimmed audio
                trimmed_audios = DataProcessing.trim_audio_per_onset(audio)
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
            print("-- current file: ", path)
            # -- librosa feature load
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")

            if DDM_OWN in path:  # 우리 데이터라면
                # -- trim first onset
                audio = DataProcessing.trim_audio_first_onset(audio)
                # -- feature extract
                feature = self.audio_to_feature(audio)
                # -- label: 드럼 종류마다 onset 여부
                label = self.get_label_detect_data(audio, path)
                data_feature_label.append([feature.tolist(), label])

        feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        return feature_df

    """
    -- detect type label 그래프
    """

    def show_detect_label_plot(self, label: List[List[float]]):
        label = np.array(label)
        for i in range(len(CODE2DRUM)):
            plt.subplot(8, 1, i + 1)
            plt.plot(label[:, i])
        plt.title("Model label")
        plt.show()

    """
    -- rhythm type feature, label 추출
    """

    def rhythm_feature_extractor(self, audio_paths: List[str]):
        data_feature_label = []

        print(f"-- ! ADT type : {self.method_type} ! --")
        print(f"-- ! {self.feature_type} feature extracting ! --")
        for path in audio_paths:
            if IDMT in path:  # IDMT data
                if "MIX" not in path:
                    continue

            print("-- current file: ", path)
            # -- librosa feature load
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")

            if DDM_OWN in path:  # 우리 데이터라면
                # -- trim first onset
                audio = DataProcessing.trim_audio_first_onset(audio)
                # -- feature extract
                feature = self.audio_to_feature(audio)
                # -- label: onset 여부
                onsets_arr = self.onset_detection.onset_detection(audio)
                label = self.get_label_rhythm_data(len(audio) / SAMPLE_RATE, onsets_arr)
                data_feature_label.append([feature.tolist(), label])
                continue

            # -- chunk
            chunk_list = DataProcessing.cut_chunk_audio(audio)
            onsets_arr = []

            if IDMT in path:  # IDMT data
                file_name = os.path.basename(path)[:-4]  # 파일 이름
                file_paths = path.split("/")[:-2]  # 뒤에서 2개 제외한 폴더 list
                xml_file = os.path.join(os.path.join(*file_paths), "annotation_xml")
                xml_file = os.path.join(xml_file, f"{file_name}.xml")
                onsets_arr = self.get_onsets_arr_from_xml(xml_file)

            if ENST in path:  # ENST data
                file_name = os.path.basename(path)[:-4]  # 파일 이름
                file_paths = path.split("/")[:-3]  # 뒤에서 3개 제외한 폴더 list
                txt_file = os.path.join(os.path.join(*file_paths), "annotation")
                txt_file = os.path.join(txt_file, f"{file_name}.txt")
                onsets_arr = self.get_onsets_arr_from_txt(txt_file)

            if E_GMD in path:  # E-GMD data
                file_name = os.path.basename(path)[:-4]  # 파일 이름
                file_paths = path.split("/")[:-1]  # 뒤에서 1개 제외한 폴더 list
                mid_file = os.path.join(os.path.join(*file_paths), f"{file_name}.mid")
                onsets_arr = self.get_onsets_arr_from_mid(mid_file)

            # -- labeling: onset 여부
            chunk_onsets_arr = self.split_onset_match_chunk(onsets_arr)
            for idx, chunk in enumerate(chunk_list):
                if not idx in chunk_onsets_arr:
                    continue

                # -- feature extract
                feature = self.audio_to_feature(chunk)
                label = self.get_label_rhythm_data(
                    len(chunk) / SAMPLE_RATE, chunk_onsets_arr[idx]
                )
                data_feature_label.append([feature.tolist(), label])

                del feature
                del label

            del chunk_list
            del onsets_arr
            del chunk_onsets_arr

        feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        if len(feature_df) > 0:
            self.show_rhythm_label_plot(feature_df.label[0])
        return feature_df

    """
    -- rhythm type label 그래프
    """

    def show_rhythm_label_plot(self, label: List[float]):
        label = np.array(label)
        plt.plot(label)
        plt.title("Model label")
        plt.show()

        # 이미지 폴더 존재 확인
        if not os.path.exists(IMAGE_PATH):
            os.mkdir(IMAGE_PATH)  # 없으면 새로 생성

        # 현재 날짜와 시간 가져오기
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{IMAGE_PATH}/{self.method_type}-label-test-{date_time}.png")

    """
    -- mel-spectrogram 그래프
    """

    def show_mel_spectrogram_plot(self, mel_spectrogram: np.ndarray):
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
        img = librosa.display.specshow(
            S_dB,
            x_axis="time",
            y_axis="mel",
            sr=SAMPLE_RATE,
            ax=ax,
            fmax=self.feature_param["fmax"],
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Mel-frequency spectrogram")
        # plt.savefig("mel-spectrogram-test.png")

    """ 
    -- method type 에 따라 feature, label 추출 후 저장
    """

    def feature_extractor(self, audio_paths: List[str]):
        features_df_origin = self.load_feature_file()  # load feature file

        features_df_new = None
        if self.method_type == METHOD_CLASSIFY:
            features_df_new = self.classify_feature_extractor(audio_paths)
        elif self.method_type == METHOD_DETECT:
            features_df_new = self.detect_feature_extractor(audio_paths)
        elif self.method_type == METHOD_RHYTHM:
            features_df_new = self.rhythm_feature_extractor(audio_paths)

        # Convert into a Panda dataframe & Add dataframe
        features_total_df = features_df_new
        if features_df_origin is not None:
            features_total_df = pd.concat(
                [features_df_origin, features_df_new], ignore_index=True
            )

        # Save feature file
        self.save_feature_file(features_total_df)
