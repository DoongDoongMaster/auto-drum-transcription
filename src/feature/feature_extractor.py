import os
import csv
import copy
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
    CLASSIFY_SAME_TIME,
    CLASSIFY_SHORT_TIME,
    CODE2DRUM,
    DATA_ENST_TEST,
    DRUM_KIT,
    E_GMD,
    E_GMD_INFO,
    ENST,
    ENST_PUB,
    FEATURE_DTYPE_16,
    FEATURE_DTYPE_32,
    IDMT,
    LABEL_COLUMN,
    LABEL_INIT_DATA,
    LABEL_DDM,
    LABEL_REF,
    LABEL_TYPE,
    MDB,
    MDB_TRAIN_SET,
    ONSET_DURATION_RIGHT,
    RAW_PATH,
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
    CLASSIFY_TYPES,
    CLASSIFY_CODE2DRUM,
    TEST,
    TRAIN,
    VALIDATION,
)


class FeatureExtractor:
    """
    데이터에서 feature를 추출하고, 라벨링하고, 저장하고, 불러오는 클래스
    """

    @staticmethod
    def load_feature_file(
        method_type: str,
        feature_type: str,
        feature_extension: str,
        data_type: str = E_GMD,  # E-GMD | IDMT | ENST
        split_type: str = TRAIN,  # TRAIN | VALIDATION | TEST
        feature_files: list[str] = None,
    ) -> pd.DataFrame:
        """
        -- feature 추출한 파일 불러오기
        """
        if (
            feature_files is None
        ):  # 피쳐 파일 리스트가 비어있다면 -> 피쳐 저장된 경로 통해서 접근
            save_folder_path = FeatureExtractor._get_save_folder_path(
                method_type, feature_type, data_type, split_type
            )

            print("--------", save_folder_path)
            if not os.path.exists(save_folder_path):
                raise Exception(
                    f"모델: {method_type}, 피쳐: {feature_type} 에 해당하는 피쳐가 없습니다!!!"
                )
            feature_files = glob(f"{save_folder_path}/*.{feature_extension}")

        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)
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

        # row 생략 없이 출력
        # pd.set_option("display.max_rows", None)
        # print("-- 추출 : ", feature_files)
        print(
            "-- ! 로딩 완료 ! --",
            "data shape:",
            combined_df.shape,
        )
        print("-- ! features ! -- ")
        print(combined_df)

        return combined_df

    @staticmethod
    def load_dataset_from_split_data_file(
        method_type: str,
        feature_type: str,
        feature_extension: str,
        split_data: dict[str],
        feature_files: list[str] = None,
    ) -> pd.DataFrame:
        """
        -- feature 추출한 파일 불러오기
        input: {train: [] / validation:[] / test:[]} 각 split type에 가져오고 싶은 데이터별로 array에 담아서
        - split_type: TRAIN | VALIDATION | TEST
        - data_type:  E-GMD | IDMT | ENST

        result_data: {train: [df, ...] / validation:[] / test:[df, ...]}
        """
        result_data = {}
        for split_type, data_type in split_data.items():
            print("-- !! split type >> ", split_type)
            print("-- !! data types >> ", data_type)
            for dt in data_type:
                combined_df = FeatureExtractor.load_feature_file(
                    method_type,
                    feature_type,
                    feature_extension,
                    dt,
                    split_type,
                    feature_files,
                )
                result_data.update({split_type: combined_df})
                del combined_df

        return result_data

    @staticmethod
    def split_train_test_from_path(audio_paths: List[str]):
        # - idmt : train / test (from path)
        # - enst : train / test (from enst test data)
        # - e-gmd : train / validation / test (from info.csv)
        # return {idmt:{train:[], test:[]}, enst:{train:[], test:[]}, e-gmd:{train:[], validation:[], test:[]}}

        # 그냥 복사하면 주소값이 복사되어, LABEL_INIT_DATA에 값이 쌓이므로, 깊은 복사로 값만 가져가기
        result_data = copy.deepcopy(LABEL_INIT_DATA)

        for path in audio_paths:
            if DataLabeling.validate_supported_data(path) == False:
                continue

            # idmt 인 경우 path에서 읽기
            if IDMT in path:
                if "WaveDrum02" in path and "train" not in path:  # -- test
                    result_data[IDMT][TEST].append(path)
                else:  # -- train
                    result_data[IDMT][TRAIN].append(path)
            # enst 인 경우 임의로 지정한 test set으로 분기
            elif ENST in path:
                dn = os.path.dirname(path)
                bn = os.path.basename(path)
                if DATA_ENST_TEST["directory"] in dn and DATA_ENST_TEST["test"] in bn:
                    result_data[ENST][TEST].append(path)
                else:
                    result_data[ENST][TRAIN].append(path)
            elif ENST_PUB in path:
                dn = os.path.dirname(path)
                bn = os.path.basename(path)
                if DATA_ENST_TEST["directory"] in dn and DATA_ENST_TEST["test"] in bn:
                    result_data[ENST_PUB][TEST].append(path)
                else:
                    result_data[ENST_PUB][TRAIN].append(path)
            # e-gmd 인 경우 csv에서 읽기
            elif E_GMD in path:
                # CSV 파일 열고 읽기 모드로 연 후, DictReader를 사용하여 데이터 읽어오기
                with open(E_GMD_INFO, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # -- ../data/raw/e-gmd-v1.0.0/ 제거 후
                        # -- audio file name과 같은 경우 train/validation/test 나눠서 데이터 추가
                        substring_to_remove = f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}/"
                        find_path = path.replace(substring_to_remove, "")
                        if find_path in row["audio_filename"]:
                            result_data[E_GMD][row["split"]].append(path)
                    del reader
            # MDB 인 경우 md에서 읽기
            elif MDB in path:
                if any(train_set in path for train_set in MDB_TRAIN_SET):
                    result_data[MDB][TRAIN].append(path)
                else:  # -- test
                    result_data[MDB][TEST].append(path)
            # drum_kit 인 경우 모두 train으로
            elif DRUM_KIT in path:
                result_data[DRUM_KIT][TRAIN].append(path)
        del audio_paths
        return result_data

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

        batch_size = 5
        for i in range(0, len(audio_paths), batch_size):
            print(f"-- ! feature extracting ... {i} to {i + batch_size}")
            batch_audio_paths = audio_paths[i : min(len(audio_paths), i + batch_size)]

            # [e-gmd/idmt/enst] : [train/validation/test] 각 데이터가 있는 경우, 각자 list로 feature extract 후, save
            split_data = FeatureExtractor.split_train_test_from_path(batch_audio_paths)
            del batch_audio_paths

            for data_type, split_data_all in split_data.items():
                for split_type, split_value in split_data_all.items():
                    print(
                        f"!! --- data_type: {data_type} -- {split_type} : {split_value}"
                    )
                    if len(split_value) != 0:
                        features_df_new = FeatureExtractor._feature_extractor(
                            split_value, method_type, feature_type
                        )
                        if features_df_new.empty:
                            continue
                        FeatureExtractor._save_feature_file(
                            method_type,
                            feature_type,
                            feature_extension,
                            features_df_new,
                            data_type,
                            split_type,
                            i,
                        )
            del split_data

    @staticmethod
    def _feature_extractor(
        audio_paths: List[str], method_type: str, feature_type: str
    ) -> pd.DataFrame:
        # dataframe 초기화
        combined_df = FeatureExtractor._init_combine_df(method_type, feature_type)

        for path in audio_paths:
            print("-- current file: ", path)

            # -- librosa feature load
            audio = FeatureExtractor.load_audio(path)

            df = FeatureExtractor._get_one_path_feature_label(
                audio, path, method_type, feature_type
            )  # 메소드별로 피쳐 & 라벨 추출
            del audio
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            del df

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
        del onsets_arr

        audios = DataProcessing.trim_audio_per_onset_with_duration(audio, onsets)
        # DataProcessing.write_trimmed_audio(
        #     "../data/classify-test", "classify_test", audios
        # )
        del audio, onsets

        for i, ao in enumerate(audios):
            feature = AudioToFeature.extract_feature(ao, method_type, feature_type)
            # make dataframe
            df = FeatureExtractor._make_dataframe(
                method_type, feature_type, feature, label[i]
            )
            # combine dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            del df, feature

        del audios
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
    def _validate_possible_label(label_dict):
        # 'KK' 키를 제외한 나머지 키에서 [1] 값을 가지는 키의 개수를 카운트합니다.
        count = sum(
            1
            for key, value in label_dict.items()
            if key != "KK" and key != "CH" and value == [1]
        )

        # [1] 값을 가지는 키의 개수가 3개 이상이면 False를 반환합니다.
        return count < 3

    @staticmethod
    def _validate_same_time_data(curr_onset, next_onset):
        return (next_onset - curr_onset) <= CLASSIFY_SAME_TIME

    @staticmethod
    def _is_available_drum_code(drum_code):
        return drum_code in CODE2DRUM

    @staticmethod
    def _get_same_time_label(idx, onsets, curr_onset, is_available, label_dict):
        if idx + 1 >= len(onsets):
            return idx, is_available, label_dict

        next_onset, next_drum = onsets[idx + 1].values()
        if FeatureExtractor._validate_same_time_data(curr_onset, next_onset):
            idx = idx + 1
            is_available = FeatureExtractor._is_available_drum_code(next_drum)
            label_dict[CODE2DRUM[next_drum]] = [1]

        return idx, is_available, label_dict

    @staticmethod
    def _get_onsets_label_from_onsets(onsets):
        idx = 0
        onsets_len = len(onsets)
        result_onsets = []  # [{"onset": onset, "duration": 다음 온셋 사이의 시간}, ...]
        result_label = []  # [{'OH':[], 'CH':[], 'TT':[], 'SD':[], 'KK':[]}, ...]
        while True:
            if idx >= onsets_len:  # onsets 길이 초과하면 종료
                break

            # 현재 onset 음
            curr_onset, curr_drum = onsets[idx].values()
            is_available = FeatureExtractor._is_available_drum_code(
                curr_drum
            )  # 우리가 다루는 악기인지 여부
            # labeling
            label_dict = {v: [0] for _, v in CODE2DRUM.items()}
            label_dict[CODE2DRUM[curr_drum]] = [1]

            # 현재 음 제외 최대 2개까지 동시에 친 음으로 판단
            for _ in range(2):
                idx, is_available, label_dict = FeatureExtractor._get_same_time_label(
                    idx, onsets, curr_onset, is_available, label_dict
                )

            idx = idx + 1

            if not is_available:
                continue

            duration = ONSET_DURATION_RIGHT
            if idx < onsets_len:
                duration = onsets[idx]["onset"] - curr_onset

            if duration < CLASSIFY_SHORT_TIME:  # 너무 짧게 잘린 데이터 버리기
                continue

            if not FeatureExtractor._validate_possible_label(
                label_dict
            ):  # 불가능한 라벨값이라면
                continue

            result_onsets.append({"onset": curr_onset, "duration": duration})
            result_label.append(label_dict)

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
            for k, v in CLASSIFY_TYPES.items():
                temp_label = []
                for drum_idx, origin_key in enumerate(v):
                    if len(temp_label) == 0:  # 초기화
                        temp_label = label[CLASSIFY_TYPES[k][drum_idx]]
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
    def convert_key_name(dict_data, key_name):
        """
        {CC:[]} -> {CC-LABEL_REF-[0.5-1-0.5]:[]}
        """
        converted_data_dict = {}
        for original_key, value in dict_data.items():
            new_key = f"{original_key}-{key_name}"
            converted_data_dict[new_key] = value
        return converted_data_dict

    @staticmethod
    def data_labeling_label_type(
        ao: np.ndarray,
        path: str,
        method_type: str,
        i: int = None,
        feature_shape: int = 0,
        hop_length: int = 0,
        label_type: str = LABEL_DDM,
    ):
        """
        data labeling 후, convert key name
        """
        label = DataLabeling.data_labeling(
            ao, path, method_type, i, feature_shape, hop_length, label_type
        )
        return FeatureExtractor.convert_key_name(label, label_type) if label else None

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

            label = {}
            for label_type_key in LABEL_TYPE:
                result_data = FeatureExtractor.data_labeling_label_type(
                    ao,
                    path,
                    method_type,
                    i,
                    feature.shape[0],
                    hop_length,
                    label_type_key,
                )
                if result_data == None:  # label 없음
                    continue
                label.update(result_data)
            if len(label.keys()) == 0:
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
            dtype=FEATURE_DTYPE_16,
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
                dtype=FEATURE_DTYPE_32,
            )

    @staticmethod
    def _save_feature_file(
        method_type: str,
        feature_type: str,
        feature_extension: str,
        features: pd.DataFrame,
        data_type: str,  # e-gmd|idmt|enst
        split_type: str,  # test|validation|test
        number: int,
    ):
        """
        -- feature 파일 저장하기
        """
        save_folder_path = FeatureExtractor._get_save_folder_path(
            method_type, feature_type, data_type, split_type
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
        -- [HH, OH, …, KK]-LABEL_DDM, [HH, OH, …, KK]-LABEL_REF
        """
        # feature에서 한 frame에 들어있는 sample 개수
        n_feature = FeatureExtractor._get_n_feature(
            feature_type, FEATURE_PARAM[method_type][feature_type]
        )

        if (
            method_type == METHOD_CLASSIFY
        ):  # label = ['CC', 'OH', 'CH', 'TT', 'SD', 'KK'] + feature
            return pd.DataFrame(columns=[v for _, v in CODE2DRUM.items()] + ["feature"])
        elif method_type == METHOD_DETECT:
            # ['OH-LABEL_DDM','CH-LABEL_DDM',..., 'KK-LABEL_REF'] + feature
            return pd.DataFrame(
                columns=LABEL_COLUMN
                + [f"{feature_type[:8]}{i + 1}" for i in range(n_feature)]
            )
        elif method_type == METHOD_RHYTHM:
            return pd.DataFrame(
                columns=["label"]
                + [f"{feature_type[:8]}{i + 1}" for i in range(n_feature)]
            )

    @staticmethod
    def _get_save_folder_path(method_type, feature_type, data_type, split_type) -> str:
        return f"{ROOT_PATH}/{PROCESSED_FEATURE}/{method_type}/{feature_type}/{data_type}/{split_type}"

    @staticmethod
    def load_audio(path):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        audio = librosa.effects.percussive(audio)
        return audio
