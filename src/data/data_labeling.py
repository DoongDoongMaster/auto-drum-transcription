import os
import numpy as np

from typing import List

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
        frame_length: int,
        hop_length: int,
    ):
        if DDM_OWN in path:
            pass

        elif IDMT in path:
            pass

        elif ENST in path:
            pass

        elif E_GMD in path:
            pass

    @staticmethod
    def get_label_ddm_classify(idx: int, path: str) -> List[int]:
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
    def get_label_ddm_detect(
        audio: np.ndarray, path: str, frame_length: int, hop_length: int
    ) -> List[List[int]]:
        """
        -- ddm own data detect type (sequence data) 라벨링
            onset position : 1
            onset position with ONSET_OFFSET : 0.5 (ONSET_OFFSET: onset position 양 옆으로 몇 개씩 붙일지)
            extra : 0
        """
        onsets_arr = OnsetDetect.onset_detection(audio)
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

            one_hot_label = DataLabeling.get_label_ddm_classify(pattern_idx, path)
            for i in range(soft_start_position, soft_end_position):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = (np.array(one_hot_label) / 2).tolist()  # ex. [0.5, 0, ...]
            labels[int(onset_position)] = one_hot_label  # ex. [1, 0, ...]

        return labels

    @staticmethod
    def get_label_rhythm_data(
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
