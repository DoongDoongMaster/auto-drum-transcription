import os
import librosa
import numpy as np
import pandas as pd

from ast import literal_eval
from datetime import datetime
from typing import List
from glob import glob

from data.data_labeling import DataLabeling
from data.data_processing import DataProcessing
from feature.audio_to_feature import AudioToFeature


from constant import (
    CODE2DRUM,
    SAMPLE_RATE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    ROOT_PATH,
    PROCESSED_FEATURE,
    CSV,
    PKL,
    FEATURE_PARAM,
)


class FeatureExtractor:
    """
    데이터에서 feature를 추출하고, 라벨링하고, 저장하고, 불러오는 클래스
    """

    @staticmethod
    def load_feature_file(
        method_type: str, feature_type: str, feature_extension: str
    ) -> pd.DataFrame:
        """
        -- feature 추출한 파일 불러오기
        """
        save_folder_path = FeatureExtractor._get_save_folder_path(
            method_type, feature_type
        )
        if not os.path.exists(save_folder_path):
            raise Exception(
                f"모델: {method_type}, 피쳐: {feature_type} 에 해당하는 피쳐가 없습니다!!!"
            )

        feature_param = FEATURE_PARAM[method_type][feature_type]

        # feature에서 한 frame에 들어있는 sample 개수
        n_feature = FeatureExtractor._get_n_feature(feature_type, feature_param)

        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(
            method_type, feature_type, n_feature
        )

        feature_files = glob(f"{save_folder_path}/*.{feature_extension}")
        for feature_file in feature_files:
            # feature 파일을 읽어와 DataFrame으로 변환
            data_feature_label = FeatureExtractor._load_feature_one_file(
                feature_file, feature_extension
            )

            # 현재 파일의 데이터를 combined_df에 추가
            combined_df = pd.concat(
                [combined_df, data_feature_label], ignore_index=True
            )
            del data_feature_label

        print(
            "-- ! 로딩 완료 ! --",
            "data shape:",
            combined_df.shape,
        )
        print("-- ! features ! -- ")
        print(combined_df)

        return combined_df

    @staticmethod
    def feature_extractor(
        audio_paths: List[str],
        method_type: str,
        feature_type: str,
        feature_extension: str,
    ):
        """
        -- method type 에 따라 feature, label 추출 후 저장
        """
        print(f"-- ! ADT type : {method_type} ! --")
        print(f"-- ! {feature_type} feature extracting ! --")
        print("-- 총 audio_paths 몇 개??? >> ", len(audio_paths))

        # feature parameter info
        feature_param = FEATURE_PARAM[method_type][feature_type]
        hop_length = feature_param.get("hop_length")

        batch_size = 20
        for i in range(0, len(audio_paths), batch_size):
            print(f"-- ! feature extracting ... {i} to {i + batch_size}")
            batch_audio_paths = audio_paths[i : min(len(audio_paths), i + batch_size)]
            features_df_new = FeatureExtractor._feature_extractor(
                batch_audio_paths, method_type, feature_type, hop_length
            )
            if features_df_new.empty:
                continue

            # Save feature file
            FeatureExtractor._save_feature_file(
                method_type, feature_type, feature_extension, features_df_new, i
            )

    @staticmethod
    def _feature_extractor(
        audio_paths: List[str],
        method_type: str,
        feature_type: str,
        hop_length: int,
    ) -> pd.DataFrame:
        # feature에서 한 frame에 들어있는 sample 개수
        n_feature = FeatureExtractor._get_n_feature(
            feature_type, FEATURE_PARAM[method_type][feature_type]
        )
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(
            method_type, feature_type, n_feature
        )
        for path in audio_paths:
            if DataLabeling.validate_supported_data(path, method_type) == False:
                continue

            print("-- current file: ", path)
            audios = FeatureExtractor._get_chunk_audio(path, method_type)
            for i, audio in enumerate(audios):
                # audio to feature
                feature = AudioToFeature.extract_feature(
                    audio, method_type, feature_type
                )
                # get label
                label = DataLabeling.data_labeling(
                    audio, path, method_type, i, feature.shape[0], hop_length
                )
                if label == False:  # label 없음
                    continue

                # make dataframe
                df = FeatureExtractor._make_dataframe(
                    method_type, feature_type, feature, label
                )
                # combine dataframe
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df

    @staticmethod
    def _make_dataframe(
        method_type: str, feature_type: str, feature: np.ndarray, label
    ) -> pd.DataFrame:
        if method_type == METHOD_CLASSIFY:
            data_feature_label = [[feature.tolist(), label]]
            df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
            return df

        if method_type in [METHOD_DETECT, METHOD_RHYTHM]:
            # 추출할 feature 개수
            n_features = FeatureExtractor._get_n_feature(
                feature_type, FEATURE_PARAM[method_type][feature_type]
            )
            # detect | rhythm 에 따라 라벨링 다르니까
            label_data = {}
            if method_type == METHOD_DETECT:
                label_data = {
                    "HH": label["HH"],
                    "ST": label["ST"],
                    "SD": label["SD"],
                    "KK": label["KK"],
                }
            elif method_type == METHOD_RHYTHM:
                label_data = {
                    "label": label,
                }
            df_meta = pd.DataFrame(
                label_data,
                dtype="float16",
            )
            df_feature = pd.DataFrame(
                feature,
                columns=[feature_type[:8] + str(i + 1) for i in range(n_features)],
                dtype="float32",
            )
            df = pd.concat([df_meta, df_feature], axis=1)  # Concatenate along columns
            return df

    @staticmethod
    def _get_chunk_audio(path: str, method_type: str) -> List[np.ndarray]:
        # -- librosa feature load
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        audio = librosa.effects.percussive(audio)

        if method_type == METHOD_CLASSIFY:
            onsets_arr = DataLabeling.get_onsets_arr(audio, path)
            return DataProcessing.trim_audio_per_onset(audio, onsets_arr)
        if method_type == METHOD_RHYTHM or method_type == METHOD_DETECT:
            return DataProcessing.cut_chunk_audio(audio)

    @staticmethod
    def _save_feature_file(
        method_type: str,
        feature_type: str,
        feature_extension: str,
        features: pd.DataFrame,
        number: int,
    ):
        """
        -- feature 파일 저장하기
        """
        save_folder_path = FeatureExtractor._get_save_folder_path(
            method_type, feature_type
        )
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        save_path = f"{save_folder_path}/{feature_type}-{date_time}-{number:04}.{feature_extension}"

        os.makedirs(save_folder_path, exist_ok=True)  # feature 폴더 생성

        if feature_extension == CSV:
            # Save csv file
            features.to_csv(save_path, sep=",")
        elif feature_extension == PKL:
            # Save pickle file
            features.to_pickle(save_path)

        print("-- ! 완료 & 새로 저장 ! --")
        print("-- ! location: ", save_path)
        print("-- ! features shape:", features.shape)

    @staticmethod
    def _load_feature_one_file(path: str, feature_extension: str) -> pd.DataFrame:
        if feature_extension == CSV:
            return pd.read_csv(
                path,
                index_col=0,
                converters={"feature": literal_eval, "label": literal_eval},
            )
        if feature_extension == PKL:
            return pd.read_pickle(path)

    @staticmethod
    def _get_n_feature(feature_type: str, feature_param: dict) -> int:
        if feature_type == STFT:
            return 1 + (feature_param["n_fft"] // 2)
        if feature_type == MFCC:
            return feature_param["n_mfcc"]
        if feature_type == MEL_SPECTROGRAM:
            return feature_param["n_mels"]

    @staticmethod
    def _init_combine_df(
        method_type: str, feature_type: str, n_feature: int
    ) -> pd.DataFrame:
        """
        -- dataframe 헤더 초기화
        """
        if method_type == METHOD_CLASSIFY:  # feature + label 형식
            return pd.DataFrame(columns=["feature", "label"])
        if method_type == METHOD_DETECT:
            return pd.DataFrame(
                columns=[v for _, v in CODE2DRUM.items()]  # ['HH', 'ST', 'SD', 'KK']
                + [feature_type[:8] + str(i + 1) for i in range(n_feature)]
            )
        if method_type == METHOD_RHYTHM:
            return pd.DataFrame(
                columns=["label"]
                + [feature_type[:8] + str(i + 1) for i in range(n_feature)]
            )

    @staticmethod
    def _get_save_folder_path(method_type, feature_type) -> str:
        return f"{ROOT_PATH}/{PROCESSED_FEATURE}/{method_type}/{feature_type}"
