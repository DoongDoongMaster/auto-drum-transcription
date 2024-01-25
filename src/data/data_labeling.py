import os
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from datetime import datetime

from data.onset_detection import OnsetDetect

from constant import (
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
        idx: int = 0,
        frame_length: int = 0,
        hop_length: int = 0,
    ):
        """
        -- method type과 data origin에 따른 data labeling 메소드
        """
        onsets_arr = DataLabeling._get_onsets_arr(audio, path, idx)

        if method_type == METHOD_CLASSIFY:
            return DataLabeling._get_label_ddm_classify(idx, path)

        if method_type == METHOD_DETECT:
            return DataLabeling._get_label_ddm_detect(
                onsets_arr, path, frame_length, hop_length
            )

        if method_type == METHOD_RHYTHM:
            return DataLabeling._get_label_rhythm_data(
                onsets_arr, frame_length, hop_length
            )

        raise Exception(f"지원하지 않는 모델 방식 {method_type} !!!")

    @staticmethod
    def validate_supported_data(path: str, method_type: str):
        # 우선 classify, detect 방식에는 ddm own data만 가능
        if method_type in [METHOD_CLASSIFY, METHOD_DETECT] and DDM_OWN not in path:
            return False
        if IDMT in path and "MIX" not in path:
            return False
        return True

    @staticmethod
    def _get_onsets_arr(audio: np.ndarray, path: str, idx: int) -> List[float]:
        start = idx * CHUNK_LENGTH  # onset 자르는 시작 초
        end = (idx + 1) * CHUNK_LENGTH  # onset 자르는 끝 초

        if DDM_OWN in path:
            return OnsetDetect.onset_detection(audio)

        if IDMT in path:
            label_path = DataLabeling._get_label_path(path, 2, "xml", "annotation_xml")
            return OnsetDetect.get_onsets_from_xml(label_path, start, end)

        if ENST in path:
            label_path = DataLabeling._get_label_path(path, 3, "txt", "annotation")
            return OnsetDetect.get_onsets_from_txt(label_path, start, end)

        if E_GMD in path:
            label_path = DataLabeling._get_label_path(path, 1, "mid")
            return OnsetDetect.get_onsets_from_mid(label_path, start, end)

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
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/label-{date_time}.png")
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
    def _get_label_ddm_classify(idx: int, path: str) -> List[int]:
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

            one_hot_label = DataLabeling._get_label_ddm_classify(pattern_idx, path)
            for i in range(soft_start_position, soft_end_position):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = (np.array(one_hot_label) / 2).tolist()  # ex. [0.5, 0, ...]
            labels[int(onset_position)] = one_hot_label  # ex. [1, 0, ...]

        return labels

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
