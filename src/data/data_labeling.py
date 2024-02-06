import os
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from datetime import datetime

from data.onset_detection import OnsetDetect

from constant import (
    DATA_ENST_NOT,
    PATTERN_DIR,
    PER_DRUM_DIR,
    PATTERN2CODE,
    ONEHOT_DRUM2CODE,
    SAMPLE_RATE,
    CODE2DRUM,
    ONSET_OFFSET,
    DDM_OWN,
    IDMT,
    ENST,
    E_GMD,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    CHUNK_LENGTH,
    IMAGE_PATH,
    DATA_ALL,
    DATA_IDMT_NOT,
    CLASSIFY_DRUM,
)


class DataLabeling:
    """
    model method type과 data origin에 따른 data labeling 관련 클래스
    """

    @staticmethod
    def data_labeling(
        audio: np.ndarray,
        path: str,
        method_type: str,
        idx: int = None,
        frame_length: int = 0,
        hop_length: int = 0,
    ):
        """
        -- method type과 data origin에 따른 data labeling 메소드
        """
        if method_type == METHOD_CLASSIFY:
            if DDM_OWN in path:
                return DataLabeling._get_ddm_single_label(idx, path)
            return DataLabeling._get_label_classify(path)

        if frame_length == 0:
            frame_length = len(audio) // hop_length

        # -- [instrument] --
        if method_type == METHOD_DETECT:
            onsets_arr = DataLabeling.get_onsets_instrument_arr(audio, path, idx)
            if DataLabeling._is_dict_all_empty(onsets_arr):
                return False

            # if DDM_OWN in path:
            #     return DataLabeling._get_ddm_multiple_label(
            #         onsets_arr, path, frame_length, hop_length
            #     )
            return DataLabeling._get_label_detect(
                onsets_arr, path, frame_length, hop_length
            )

        # -- [only onset] --
        if method_type == METHOD_RHYTHM:
            onsets_arr = DataLabeling.get_onsets_arr(audio, path, idx)
            if len(onsets_arr) == 0:
                return False
            return DataLabeling._get_label_rhythm_data(
                onsets_arr, frame_length, hop_length
            )

        raise Exception(f"지원하지 않는 모델 방식 {method_type} !!!")

    @staticmethod
    def validate_supported_data(path: str, method_type: str):
        # 우리가 사용할 데이터 형태 아닌 경우
        if not any(p in path for p in DATA_ALL):
            return False
        # IDMT: train 들어가면 x
        if IDMT in path and any(p in path for p in DATA_IDMT_NOT):
            return False
        # ENST: accompaniment 들어가면 x
        if ENST in path and any(p in path for p in DATA_ENST_NOT):
            return False
        return True

    @staticmethod
    def get_onsets_arr(audio: np.ndarray, path: str, idx: int = None) -> List[float]:
        if idx is None:  # idx 없다면 처음부터 끝까지의 onset을 구함
            start = 0
            end = None
        else:
            start = idx * CHUNK_LENGTH  # onset 자르는 시작 초
            end = (idx + 1) * CHUNK_LENGTH  # onset 자르는 끝 초

        if DDM_OWN in path:
            return OnsetDetect.onset_detection(audio)

        if IDMT in path:
            if "MIX" in path:
                label_path = DataLabeling._get_label_path(
                    path, 2, "xml", "annotation_xml"
                )
                return OnsetDetect.get_onsets_from_xml(label_path, start, end)
            else:
                label_path = DataLabeling._get_label_path(
                    path, 2, "svl", "annotation_svl"
                )
                return OnsetDetect.get_onsets_from_svl(label_path, start, end)

        if ENST in path:
            label_path = DataLabeling._get_label_path(path, 3, "txt", "annotation")
            return OnsetDetect.get_onsets_from_txt(label_path, start, end)

        if E_GMD in path:
            label_path = DataLabeling._get_label_path(path, 1, "mid")
            return OnsetDetect.get_onsets_from_mid(label_path, start, end)

    @staticmethod
    def get_onsets_instrument_arr(
        audio: np.ndarray, path: str, idx: int = None
    ) -> List[float]:
        if idx is None:  # idx 없다면 처음부터 끝까지의 onset을 구함
            start = 0
            end = None
        else:
            start = idx * CHUNK_LENGTH  # onset 자르는 시작 초
            end = (idx + 1) * CHUNK_LENGTH  # onset 자르는 끝 초

        # {'HH':[], 'ST':[], 'SD':[], 'HH':[]}
        label_init = {v: [] for _, v in CODE2DRUM.items()}
        label = label_init

        # if DDM_OWN in path:
        #     # return OnsetDetect.onset_detection(audio)
        #     label = {}

        if IDMT in path:
            if "MIX" in path:
                label_path = DataLabeling._get_label_path(
                    path, 2, "xml", "annotation_xml"
                )
                label = OnsetDetect.get_onsets_instrument_from_xml(
                    label_path, start, end, label_init
                )
            else:
                label_path = DataLabeling._get_label_path(
                    path, 2, "svl", "annotation_svl"
                )
                label = OnsetDetect.get_onsets_instrument_from_svl(
                    label_path, start, end, label_init
                )

        if ENST in path:
            label_path = DataLabeling._get_label_path(path, 3, "txt", "annotation")
            label = OnsetDetect.get_onsets_instrument_from_txt(
                label_path, start, end, label_init
            )

        # if E_GMD in path:
        #     label_path = DataLabeling._get_label_path(path, 1, "mid")
        #     # return OnsetDetect.get_onsets_from_mid(label_path, start, end)
        #     label = {}

        return label

    @staticmethod
    def show_label_plot(label):
        """
        -- label 그래프
        """
        data = np.array(label)
        data = data.reshape(data.shape[0], -1)

        for i in range(data.shape[1]):
            plt.subplot(data.shape[1], 1, i + 1)
            plt.plot(data[:, i])

        plt.title("Model Label")
        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/label-{date_time}.png")
        plt.show()

    @staticmethod
    def show_label_onset_plot(label: List[float], onset: List[int]):
        """
        -- label, onset 그래프
        """
        data = np.array(label)
        data = data.reshape(data.shape[0], -1)

        for i in range(data.shape[1]):
            plt.subplot(data.shape[1], 1, i + 1)
            plt.plot(data[:, i])

        plt.plot(onset, data[onset], "x")
        plt.title("Model Label")
        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/label-onset-{date_time}.png")
        plt.show()

    @staticmethod
    def _get_label_path(
        audio_path: str, back_move_num: int, extension: str, folder_name: str = ""
    ) -> str:
        """
        -- label file의 path를 audio path로부터 구하는 함수
        """
        file_name = os.path.basename(audio_path)[:-4]  # 파일 이름
        file_paths = audio_path.split("/")[
            :-back_move_num
        ]  # 뒤에서 back_move_num 개 제외한 폴더 list
        label_file = os.path.join(os.path.join(*file_paths), folder_name)
        label_file = os.path.join(label_file, f"{file_name}.{extension}")
        return label_file

    @staticmethod
    def _get_ddm_single_label(idx: int, path: str) -> List[int]:
        """
        -- ddm own data classify type (trimmed data) 라벨링
        """
        file_name = os.path.basename(path)  # extract file name
        if PATTERN_DIR in path:  # -- pattern
            pattern_name = file_name[:2]  # -- P1
            label = PATTERN2CODE[pattern_name][idx]
        elif PER_DRUM_DIR in path:  # -- per drum
            drum_name = file_name[:2]  # -- CC
            label = ONEHOT_DRUM2CODE[drum_name]
        return label

    @staticmethod
    def _get_ddm_multiple_label(idx: int, path: str) -> List[int]:
        """
        -- ddm own data classify type (trimmed data) 라벨링
        """
        label = {}
        """
        1
        HH [1,0,0,0]

        {hh:[1], sd:[0]}


        2
        HH
        SD

        {hh:[1,1], sd:[0,1]}
        
        """

        return label

    @staticmethod
    def _get_label_classify(path: str):
        for idx, words in CLASSIFY_DRUM.items():
            if any((w in path) for w in words):
                return ONEHOT_DRUM2CODE[CODE2DRUM[idx]]

    @staticmethod
    def _get_frame_index(time: float, hop_length: int) -> int:
        """
        -- hop length 기반으로 frame의 인덱스 구하는 함수
        """
        return int(time * SAMPLE_RATE / float(hop_length))

    @staticmethod
    def _get_label_ddm_detect(
        onsets_arr: List[float], path: str, frame_length: int, hop_length: int
    ) -> List[List[int]]:
        """
        -- ddm own data detect type (sequence data) 라벨링
            onset position : 1
            onset position with ONSET_OFFSET : 0.5 (ONSET_OFFSET: onset position 양 옆으로 몇 개씩 붙일지)
            extra : 0
        """
        labels = [[0] * len(CODE2DRUM) for _ in range(frame_length)]

        for pattern_idx, onset in enumerate(onsets_arr):
            onset_position = DataLabeling._get_frame_index(onset, hop_length)
            if onset_position >= frame_length:
                break

            soft_start_position = max(  # -- onset - offset
                (onset_position - ONSET_OFFSET), 0
            )
            soft_end_position = min(  # -- onset + offset
                onset_position + ONSET_OFFSET + 1, frame_length
            )

            one_hot_label = DataLabeling._get_ddm_single_label(pattern_idx, path)
            for i in range(soft_start_position, soft_end_position):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = (np.array(one_hot_label) / 2).tolist()  # ex. [0.5, 0, ...]
            labels[int(onset_position)] = one_hot_label  # ex. [1, 0, ...]

        return labels

    @staticmethod
    def _get_label_detect(
        onsets_arr: List[float], path: str, frame_length: int, hop_length: int
    ) -> List[List[int]]:
        label = {v: [] for _, v in CODE2DRUM.items()}

        """
        각 HH, SD... 마다 _get_label_rhythm_data 해서 라벨링
        """
        for drum_type, onset_times in onsets_arr.items():
            drum_label = DataLabeling._get_label_rhythm_data(
                onset_times, frame_length, hop_length
            )
            label[drum_type] = drum_label

        return label

    @staticmethod
    def _get_label_rhythm_data(
        onsets_arr: List[float], frame_length: int, hop_length: int
    ) -> List[float]:
        """
        -- onset 라벨링 (ONSET_OFFSET: onset position 양 옆으로 몇 개씩 붙일지)
        """
        labels = [0] * frame_length

        for onset in onsets_arr:
            onset_position = DataLabeling._get_frame_index(onset, hop_length)  # -- 1
            if onset_position >= frame_length:
                break

            soft_start_position = max(  # -- onset - offset
                (onset_position - ONSET_OFFSET), 0
            )
            soft_end_position = min(  # -- onset + offset
                onset_position + ONSET_OFFSET + 1, frame_length
            )

            # offset -> 양 옆으로 0.5 몇 개 붙일지
            for i in range(soft_start_position, soft_end_position):
                if labels[i] == 1:
                    continue
                labels[i] = 0.5

            labels[onset_position] = 1

        return labels

    @staticmethod
    def _is_dict_all_empty(dict_arr):
        """
        딕셔너리의 모든 value가 비어있는 지 확인하는 함수
        """
        return all(len(value) == 0 for value in dict_arr.values())
