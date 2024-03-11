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
    CLASSIFY_DURATION,
    CLASSIFY_DETECT_TYPES,
    CLASSIFY_MAP,
    CLASSIFY_DRUM2CODE,
    CLASSIFY_CODE2DRUM,
    CLASSIFY_IMPOSSIBLE_LABEL,
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

        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

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

        batch_size = 20
        for i in range(0, len(audio_paths), batch_size):
            print(f"-- ! feature extracting ... {i} to {i + batch_size}")
            batch_audio_paths = audio_paths[i : min(len(audio_paths), i + batch_size)]
            features_df_new = FeatureExtractor._feature_extractor(
                batch_audio_paths, method_type, feature_type
            )
            if features_df_new.empty:
                continue

            # Save feature file
            FeatureExtractor._save_feature_file(
                method_type, feature_type, feature_extension, features_df_new, i
            )

    @staticmethod
    def _feature_extractor(
        audio_paths: List[str], method_type: str, feature_type: str
    ) -> pd.DataFrame:
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

        for path in audio_paths:
            if DataLabeling.validate_supported_data(path) == False:
                continue

            print("-- current file: ", path)

            # -- librosa feature load
            audio = FeatureExtractor.load_audio(path)

            df = FeatureExtractor._get_one_path_feature_label(
                audio, path, method_type, feature_type
            )  # 메소드별로 피쳐 & 라벨 추출
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df

    @staticmethod
    def _get_one_path_feature_label(
        audio: np.ndarray, path: str, method_type: str, feature_type: str
    ):
        """
        하나의 path에 대한 feature와 label을 구하는 함수 (method type에 따라)
        """
        if method_type == METHOD_CLASSIFY:
            return FeatureExtractor._extract_classify_feature_per_onsets(
                audio, path, method_type, feature_type
            )
        if method_type in [METHOD_DETECT, METHOD_RHYTHM]:
            return FeatureExtractor._extract_non_classify_feature(
                audio, path, method_type, feature_type
            )

    @staticmethod
    def _extract_classify_feature_per_onsets(
        audio: np.ndarray, path: str, method_type: str, feature_type: str
    ):
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

        # onsets 초 별로 악기 있는 형태로 구하기
        onsets_arr = DataLabeling.get_onsets_instrument_all_arr(audio, path)

        # 동시에 친 데이터로 치는 오프셋만큼 묶으면서 데이터 라벨링
        onsets, label = FeatureExtractor._get_onsets_label_from_onsets(onsets_arr)

        audios = DataProcessing.trim_audio_per_onset_with_duration(audio, onsets)
        # DataProcessing.write_trimmed_audio("../data/test", "classify_test", audios)

        for i, ao in enumerate(audios):
            feature = AudioToFeature.extract_feature(ao, method_type, feature_type)
            # make dataframe
            df = FeatureExtractor._make_dataframe(
                method_type, feature_type, feature, label[i]
            )
            # combine dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df

    @staticmethod
    def binary_to_string(binary_list):
        # 이진수를 문자열로 변환하는 함수 정의
        binary_list = list(map(int, binary_list))  # 정수로 변환
        return "".join(map(str, binary_list))

    @staticmethod
    def binary_to_decimal(binary_string):
        # 이진수를 10진수로 변환하는 함수 정의
        return int(binary_string, 2)

    @staticmethod
    def decimal_to_binary(decimal_number):
        # 10진수를 이진수로 변환하는 함수 정의
        binary_string = bin(decimal_number)[2:]
        # 2진수를 라벨 개수 자리로 맞추기 위해 앞에 0을 채움
        binary_string = (
            "0" * (len(CLASSIFY_CODE2DRUM) - len(binary_string)) + binary_string
        )
        return [*map(int, binary_string)]

    @staticmethod
    def one_hot_label_to_number(labels: np.array):
        # 각 리스트를 이진수로 변환한 뒤 10진수로 변환하여 저장
        return np.apply_along_axis(
            lambda x: FeatureExtractor.binary_to_decimal(
                FeatureExtractor.binary_to_string(x)
            ),
            axis=1,
            arr=labels,
        )

    @staticmethod
    def number_to_one_hot_label(labels: np.array):
        # 10진수를 다시 이진수로 변환하여 배열에 저장
        return np.array(
            [FeatureExtractor.decimal_to_binary(decimal) for decimal in labels]
        )

    @staticmethod
    def _translate_drum_label_to_classify(drum: int) -> int:
        if drum == -1:
            return
        return CLASSIFY_DRUM2CODE[CLASSIFY_MAP[CODE2DRUM[drum]]]

    @staticmethod
    def _get_onsets_label_from_onsets(onsets):
        OFFSET = 0.035  # 몇 초 차이까지 동시에 친 것으로 볼 것인지
        idx = 0
        result_onsets = []  # [{"onset": onset, "duration": 다음 온셋 사이의 시간}, ...]
        result_label = []  # [{'OH':[], 'CH':[], 'TT':[], 'SD':[], 'KK':[]}, ...]
        while True:
            if idx >= len(onsets):
                break

            is_available = True  # 우리가 다루는 악기인지 여부
            curr_onset, curr_drum = onsets[idx].values()
            if not curr_drum in CODE2DRUM:
                is_available = False

            curr_drum = FeatureExtractor._translate_drum_label_to_classify(
                curr_drum
            )  # 0: 'OH', 1: 'CH', 2: 'TT', 3: 'SD', 4: 'KK'
            temp_label = [curr_drum]
            if (
                idx + 1 < len(onsets)
                and onsets[idx + 1]["onset"] - curr_onset <= OFFSET
            ):
                idx = idx + 1
                if not onsets[idx]["drum"] in CODE2DRUM:
                    is_available = False
                next_drum = FeatureExtractor._translate_drum_label_to_classify(
                    onsets[idx]["drum"]
                )  # 0: 'OH', 1: 'CH', 2: 'TT', 3: 'SD', 4: 'KK'
                temp_label.append(next_drum)
                if (
                    idx + 1 < len(onsets)
                    and onsets[idx + 1]["onset"] - curr_onset <= OFFSET
                ):
                    idx = idx + 1
                    if not onsets[idx]["drum"] in CODE2DRUM:
                        is_available = False
                    next_drum = FeatureExtractor._translate_drum_label_to_classify(
                        onsets[idx]["drum"]
                    )  # 0: 'OH', 1: 'CH', 2: 'TT', 3: 'SD', 4: 'KK'
                    temp_label.append(next_drum)

            idx = idx + 1

            if not is_available:
                continue

            duration = CLASSIFY_DURATION
            if idx < len(onsets):
                duration = onsets[idx]["onset"] - curr_onset

            if duration < 0.05:  # 너무 짧게 잘린 데이터 버리기
                continue

            label = {v: [0] for _, v in CLASSIFY_CODE2DRUM.items()}
            binary_label = [0] * len(CLASSIFY_CODE2DRUM)
            for code in temp_label:
                label[CLASSIFY_CODE2DRUM[code]] = [1]
                binary_label[code] = 1
            binary_label = [binary_label]

            if (
                FeatureExtractor.one_hot_label_to_number(np.array(binary_label))[0]
                in CLASSIFY_IMPOSSIBLE_LABEL
            ):  # 불가능한 라벨값이라면
                continue

            result_onsets.append({"onset": curr_onset, "duration": duration})
            result_label.append(label)

        # print("result_onsets-------------")
        # for idx, result in enumerate(result_onsets):
        #     print(idx + 1, result["duration"])
        # print("result_label-------------")
        # print(result_label)

        return result_onsets, result_label

    @staticmethod
    def _extract_classify_feature_per_frame(
        audio: np.ndarray, path: str, method_type: str, feature_type: str
    ):
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

        # feature parameter info
        feature_param = FEATURE_PARAM[method_type][feature_type]
        hop_length = feature_param.get("hop_length")

        # audio 전체 길이에 대한 label 구하기
        label = DataLabeling.data_labeling(
            audio, path, method_type, hop_length=hop_length
        )

        onsets_frame = FeatureExtractor._label_to_onset_frame(label)
        onsets_time = librosa.frames_to_time(
            onsets_frame, sr=SAMPLE_RATE, hop_length=hop_length
        )

        audios = DataProcessing.trim_audio_per_onset(audio, onsets_time)
        # DataProcessing.write_trimmed_audio("../data/test", "classify_test", audios)

        for _, ao in enumerate(audios):
            feature = AudioToFeature.extract_feature(ao, method_type, feature_type)
            l = {}
            for k, v in CLASSIFY_DETECT_TYPES.items():
                temp_label = []
                for drum_idx, origin_key in enumerate(v):
                    if len(temp_label) == 0:  # 초기화
                        temp_label = label[CLASSIFY_DETECT_TYPES[k][drum_idx]]
                    else:
                        for frame_idx, frame_value in enumerate(label[origin_key]):
                            if temp_label[frame_idx] == 1.0 or frame_value == 0.0:
                                continue
                            temp_label[frame_idx] = frame_value
                l[k] = temp_label

            print("------classify label------------", l)
            # make dataframe
            df = FeatureExtractor._make_dataframe(method_type, feature_type, feature, l)
            # combine dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df

    def _label_to_onset_frame(label: dict[str, List[int]]):
        """
        각 악기별 label 정보에서 onset frame 을 구하는 함수 (in classify)
        """
        onset_frame = []

        for j in range(
            len(label[list(label.keys())[0]])
        ):  # frame number, 첫 번째 value에 접근
            is_onset = False  # 1 인 i가 하나라도 있다면 onset으로 판단
            for key in label.keys():  # instrument number
                if label[key][j] == 1:
                    is_onset = True
                    break
            if is_onset:
                onset_frame.append(j)

        return onset_frame

    @staticmethod
    def _extract_non_classify_feature(
        audio: np.ndarray, path: str, method_type: str, feature_type: str
    ):
        """
        detect, rhythm에서의 feature와 label 구하기 (하나의 path에서)
        """
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

        # feature parameter info
        feature_param = FEATURE_PARAM[method_type][feature_type]
        hop_length = feature_param.get("hop_length")

        # 오디오 자르기
        audios = DataProcessing.cut_chunk_audio(audio)

        for i, ao in enumerate(audios):
            # audio to feature
            feature = AudioToFeature.extract_feature(ao, method_type, feature_type)
            # get label
            label = DataLabeling.data_labeling(
                ao, path, method_type, i, feature.shape[0], hop_length
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
        method_type: str,
        feature_type: str,
        feature: np.ndarray,
        label: dict[str, List[int]],
    ) -> pd.DataFrame:
        label_df = FeatureExtractor._make_label_dataframe(method_type, label)
        feature_df = FeatureExtractor._make_feature_dataframe(
            method_type, feature_type, feature
        )
        df = pd.concat([label_df, feature_df], axis=1)  # Concatenate along columns
        return df

    @staticmethod
    def _make_label_dataframe(
        method_type: str, label: dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        method type에 따라 dataframe의 label을 만드는 함수
        """
        label_data = {}
        if method_type in [METHOD_CLASSIFY, METHOD_DETECT]:
            label_data = label
        if method_type == METHOD_RHYTHM:
            label_data = {"label": label}

        return pd.DataFrame(
            label_data,
            dtype="float16",
        )

    @staticmethod
    def _make_feature_dataframe(
        method_type: str, feature_type: str, feature: np.ndarray
    ):
        """
        method type별로 feature의 dataframe을 만드는 함수
        """
        if method_type == METHOD_CLASSIFY:
            return pd.DataFrame([[feature.tolist()]], columns=["feature"])
        if method_type in [METHOD_DETECT, METHOD_RHYTHM]:
            # 추출할 feature 개수
            n_feature = FeatureExtractor._get_n_feature(
                feature_type, FEATURE_PARAM[method_type][feature_type]
            )
            return pd.DataFrame(
                feature,
                columns=[feature_type[:8] + str(i + 1) for i in range(n_feature)],
                dtype="float32",
            )

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
    def _init_combine_df(method_type: str, feature_type: str) -> pd.DataFrame:
        """
        -- dataframe 헤더 초기화
        """
        # feature에서 한 frame에 들어있는 sample 개수
        n_feature = FeatureExtractor._get_n_feature(
            feature_type, FEATURE_PARAM[method_type][feature_type]
        )

        if (
            method_type == METHOD_CLASSIFY
        ):  # label = ['OH', 'CH', 'TT', 'SD', 'KK'] + feature
            return pd.DataFrame(
                columns=[key for key in CLASSIFY_DETECT_TYPES.keys()] + ["feature"]
            )
        if method_type == METHOD_DETECT:
            return pd.DataFrame(
                columns=[
                    v for _, v in CODE2DRUM.items()
                ]  # ['CC', 'OH', 'CH', 'TT', 'SD', 'KK']
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

    @staticmethod
    def load_audio(path):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        audio = librosa.effects.percussive(audio)
        return audio
