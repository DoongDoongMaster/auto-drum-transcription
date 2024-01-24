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

from data.data_labeling import DataLabeling
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
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
    PROCESSED_FEATURE,
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


class FeatureExtractor:
    """
    데이터에서 feature를 추출하고, 라벨링하고, 저장하고, 불러오는 클래스
    """

    @staticmethod
    def load_feature_file(method_type: str, feature_type: str):
        """
        -- feature 추출한 파일 불러오기
        """
        save_folder_path = (
            f"{ROOT_PATH}/{PROCESSED_FEATURE}/{method_type}/{feature_type}/"
        )

        if not os.path.exists(save_folder_path):
            raise Exception(f"모델: {method_type}, 피쳐: {feature_type} 에 해당하는 피쳐가 없습니다!!!")

        feature_param = FEATURE_PARAM[method_type][feature_type]

        combined_df = pd.DataFrame(
            columns=["label"] + ["mel-spec" + str(i + 1) for i in range(128)]
        )

        if os.path.exists(save_folder_path):
            pkl_files = glob(f"{save_folder_path}/*.pkl")
            for pkl_file in pkl_files:
                # pkl 파일을 읽어와 DataFrame으로 변환합니다.
                data_feature_label = pd.read_pickle(pkl_file)
                # print(">>>>>>>>>>>>>>>data_feature_label", data_feature_label.head)

                # 현재 파일의 데이터를 combined_df에 추가합니다.
                combined_df = pd.concat(
                    [combined_df, data_feature_label], ignore_index=True
                )

            # for feature_file in feature_file_list:
            #     if os.path.exists(
            #         save_folder_path + feature_file
            #     ):  # 추출된 feature 존재 한다면
            #         # print("-- ! 기존 feature loading : ", self.save_path)

            #         # if self.feature_extension == CSV:
            #         #     data_feature_label = pd.read_csv(
            #         #         self.save_path,
            #         #         index_col=0,
            #         #         converters={"feature": literal_eval, "label": literal_eval},
            #         #     )
            #         if self.feature_extension == PKL:
            #             data_feature_label = pd.read_pickle(self.save_path)

        print(
            "-- ! 로딩 완료 ! --",
            "data shape:",
            combined_df.shape,
        )
        print("-- ! features ! -- ")
        print(combined_df)

        return combined_df

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

    def save_feature_file(self, features: pd.DataFrame, number):
        if self.feature_extension == CSV:
            # Save csv file
            features.to_csv(self.save_path, sep=",")
        elif self.feature_extension == PKL:
            # Save pickle file
            features.to_pickle(
                f"{self.data_root_path}/{self.method_type}/{self.feature_type}/{self.feature_type}-{number}.{self.feature_extension}"
            )

        print("-- ! 완료 & 새로 저장 ! --")
        print("-- ! location: ", self.save_path)
        print("-- ! features shape:", features.shape)

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

            # if DDM_OWN in path:  # 우리 데이터라면
            #     # -- trim first onset
            #     audio = self.data_processing.trim_audio_first_onset(audio)
            #     # -- feature extract
            #     feature = self.audio_to_feature(audio)
            #     # -- label: onset 여부
            #     onsets_arr = self.onset_detection.onset_detection(audio)
            #     label = self.get_label_rhythm_data(
            #         len(audio) / self.sample_rate, onsets_arr
            #     )
            #     data_feature_label.append([feature.tolist(), label])
            #     continue

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
                meta_data = {
                    "label": label,
                }

                df_meta = pd.DataFrame(meta_data)
                # mel-spectrogram feature size: 128
                df_mel_spec = pd.DataFrame(
                    np.transpose(feature),
                    columns=["mel-spec" + str(i + 1) for i in range(128)],
                )
                df = pd.concat(
                    [df_meta, df_mel_spec], axis=1
                )  # Concatenate along columns
                data_feature_label.append(df)

                # data_feature_label.append([feature.tolist(), label])

        # feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        feature_df = pd.concat(data_feature_label, ignore_index=True)

        if len(feature_df) > 0:
            self.show_rhythm_label_plot(feature_df.label[0])
        return feature_df

    """ 
    -- method type 에 따라 feature, label 추출 후 저장
    """

    def feature_extractor(self, audio_paths: List[str]):
        print("-- 총 audio_paths 몇 개??? >> ", len(audio_paths))

        batch_size = 20
        for i in range(0, len(audio_paths), batch_size):
            features_df_new = None
            batch_audio_paths = audio_paths[i : min(len(audio_paths), i + batch_size)]

            if self.method_type == METHOD_CLASSIFY:
                features_df_new = self.classify_feature_extractor(batch_audio_paths)
            elif self.method_type == METHOD_DETECT:
                features_df_new = self.detect_feature_extractor(batch_audio_paths)
            elif self.method_type == METHOD_RHYTHM:
                features_df_new = self.rhythm_feature_extractor(batch_audio_paths)

            # Save feature file
            self.save_feature_file(features_df_new, i)
