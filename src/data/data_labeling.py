import os
import numpy as np
import pretty_midi
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
    DRUM2CODE,
    CODE2DRUM,
    ONSET_OFFSET,
    DDM_OWN,
    IDMT,
    ENST,
    E_GMD,
    DRUM_KIT,
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
        if frame_length == 0:
            frame_length = len(audio) // hop_length

        # -- [instrument] --
        if method_type in [METHOD_CLASSIFY, METHOD_DETECT]:
            onsets_arr = DataLabeling.get_onsets_instrument_arr(audio, path, idx)
            if DataLabeling._is_dict_all_empty(onsets_arr):
                return False
            return DataLabeling._get_label_detect(onsets_arr, frame_length, hop_length)

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
    def validate_supported_data(path: str):
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
        start, end = DataLabeling._calculate_start_end(idx)

        if DDM_OWN in path:
            return OnsetDetect.get_onsets_using_librosa(audio, start, end)

        if DRUM_KIT in path:
            onsets = OnsetDetect.get_onsets_using_librosa(audio, start, end)
            return [onsets[0]]

        # label path 구하기
        label_path = DataLabeling._get_label_path_by_audio_path(path)

        onset_detection_methods = {
            IDMT: (
                OnsetDetect.get_onsets_from_xml
                if "MIX" in path
                else OnsetDetect.get_onsets_from_svl
            ),
            ENST: OnsetDetect.get_onsets_from_txt,
            E_GMD: OnsetDetect.get_onsets_from_mid,
        }

        data_own = path.split("/")[3]
        if data_own in onset_detection_methods:
            print("[", data_own, "] onset method: ", onset_detection_methods[data_own])
            return onset_detection_methods[data_own](label_path, start, end)

        raise Exception(f"지원하지 않는 데이터 {path} !!!")

    @staticmethod
    def get_onsets_instrument_arr(
        audio: np.ndarray, path: str, idx: int = None
    ) -> List[float]:
        start, end = DataLabeling._calculate_start_end(idx)

        # {'HH':[], 'ST':[], 'SD':[], 'HH':[]}
        label_init = {v: [] for _, v in CODE2DRUM.items()}
        label = label_init

        if DDM_OWN in path:
            if PER_DRUM_DIR in path:
                label = OnsetDetect.get_onsets_instrument_from_wav(
                    audio, path, start, end, label_init
                )
            # elif PATTERN_DIR in path:
            #     label_path = DataLabeling._get_ddm_label_path(
            #         path, 3, "txt", "annotation"
            #     )

        if DRUM_KIT in path:
            label = OnsetDetect.get_onsets_instrument_from_wav(
                audio, path, start, end, label_init
            )

        # label path 구하기
        label_path = DataLabeling._get_label_path_by_audio_path(path)

        onset_detection_methods = {
            IDMT: (
                OnsetDetect.get_onsets_instrument_from_xml
                if "MIX" in path
                else OnsetDetect.get_onsets_instrument_from_svl
            ),
            ENST: OnsetDetect.get_onsets_instrument_from_txt,
            E_GMD: OnsetDetect.get_onsets_instrument_from_mid,
        }

        data_own = path.split("/")[3]
        if data_own in onset_detection_methods:
            print("[", data_own, "] onset method: ", onset_detection_methods[data_own])
            label = onset_detection_methods[data_own](
                label_path, start, end, label_init
            )

        return label

    @staticmethod
    def get_onsets_instrument_all_arr(audio: np.ndarray, path: str) -> List[float]:
        result = []
        if DDM_OWN in path or DRUM_KIT in path or IDMT in path:
            label_dict = DataLabeling.get_onsets_instrument_arr(audio, path)
            result = DataLabeling._onsets_instrument_to_onsets_arr(label_dict)

        # label path 구하기
        label_path = DataLabeling._get_label_path_by_audio_path(path)

        onset_detection_methods = {
            ENST: OnsetDetect.get_onsets_instrument_all_from_txt,
            E_GMD: OnsetDetect.get_onsets_instrument_all_from_mid,
        }

        data_own = path.split("/")[3]
        if data_own in onset_detection_methods:
            print("[", data_own, "] onset method: ", onset_detection_methods[data_own])
            result = onset_detection_methods[data_own](label_path)

        sorted_result = sorted(result, key=lambda data: (data["onset"], data["drum"]))
        # print("-- ! {onset, drum} data ! --")
        # print(sorted_result)
        return sorted_result

    @staticmethod
    def show_label_plot(label: List[List[float]]):
        """
        -- label 그래프
        [[1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         ...]
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
    def show_label_dict_plot(label: dict[str, List[float]], start=0, end=None):
        """
        -- label 그래프
        {
            "HH": [1, 0, 0, ...],
            "ST": [0, 0, 0, ...],
            ...
        }
        """
        if end is None:  # end가 none이라면 y_true 끝까지
            end = len(label[CODE2DRUM[0]])

        leng = len(label.keys())
        for key, label_arr in label.items():
            data = np.array(label_arr)
            plt.subplot(leng, 1, DRUM2CODE[key] + 1)
            plt.plot(data)
            plt.axis([start, end, 0, 1])

        plt.title("Model Label")
        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/label-{date_time}.png")
        plt.show()

    @staticmethod
    def show_label_dict_compare_plot(
        y_true: dict[str, List[float]],
        y_pred: dict[str, List[float]],
        start=0,
        end=None,
    ):
        """
        -- label 그래프
        {
            "HH": [1, 0, 0, ...],
            "ST": [0, 0, 0, ...],
            ...
        }
        """
        if end is None:  # end가 none이라면 y_true 끝까지
            end = len(y_true[CODE2DRUM[0]])
        leng = len(y_true.keys()) * 2
        for key, label_arr in y_true.items():
            true_data = np.array(label_arr)
            plt.subplot(leng, 1, 2 * DRUM2CODE[key] + 1)
            plt.plot(true_data, color="b")
            plt.axis([start, end, 0, 1])

            pred_data = np.array(y_pred[key])
            plt.subplot(leng, 1, 2 * DRUM2CODE[key] + 2)
            plt.plot(pred_data, color="r")

            plt.axis([start, end, 0, 1])

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
    def _onsets_instrument_to_onsets_arr(
        onsets_instrument: dict[str, List[float]]
    ) -> List[dict[float, str]]:
        """
        -- 악기 별 onsets 배열이 있는 dictionary를 받아서 {"onset": onset, "drum": 드럼 번호} list 배열로 반환하는 함수
        """
        result = []
        for drum, onsets_arr in onsets_instrument.items():
            for onset in onsets_arr:
                data = {"onset": onset, "drum": DRUM2CODE[drum]}
                result.append(data)
        return result

    @staticmethod
    def _calculate_start_end(idx: int = None):
        start, end = 0, None  # idx 없다면 처음부터 끝까지의 onset을 구함
        if idx is not None:
            start = idx * CHUNK_LENGTH  # onset 자르는 시작 초
            end = (idx + 1) * CHUNK_LENGTH  # onset 자르는 끝 초
        return start, end

    @staticmethod
    def _get_label_path_by_audio_path(audio_path: str) -> str:
        """
        -- audio path 별로 label path를 구하는 함수
        """
        if IDMT in audio_path:
            if "MIX" in audio_path:
                return DataLabeling._get_label_path(
                    audio_path, 2, "xml", "annotation_xml"
                )
            return DataLabeling._get_label_path(audio_path, 2, "svl", "annotation_svl")

        if ENST in audio_path:
            return DataLabeling._get_label_path(audio_path, 3, "txt", "annotation")

        if E_GMD in audio_path:
            label_path = ""
            try:
                label_path = DataLabeling._get_label_path(audio_path, 1, "mid")
                midi_data = pretty_midi.PrettyMIDI(label_path)
            except:
                label_path = DataLabeling._get_label_path(audio_path, 1, "midi")
            return label_path

    @staticmethod
    def _get_label_path(
        audio_path: str, back_move_num: int, extension: str, folder_name: str = ""
    ) -> str:
        """
        -- label file의 path를 audio path로부터 구하는 함수
        """
        file_name = os.path.basename(audio_path)[:-4]  # 파일 이름
        # 뒤에서 back_move_num 개 제외한 폴더 list
        file_paths = audio_path.split("/")[:-back_move_num]
        label_file = DataLabeling._get_label_file(
            file_name, file_paths, extension, folder_name
        )
        return label_file

    @staticmethod
    def _get_ddm_label_path(
        audio_path: str, back_move_num: int, extension: str, folder_name: str = ""
    ) -> str:
        """
        (DDM용)
        -- label file의 path를 audio path로부터 구하는 함수
        """
        split_path = audio_path.split("/")
        file_name = split_path[-3]  # 파일 이름 (P1, P2,...)
        # 뒤에서 back_move_num 개 제외한 폴더 list
        file_paths = split_path[:-back_move_num]
        label_file = DataLabeling._get_label_file(
            file_name, file_paths, extension, folder_name
        )
        return label_file

    @staticmethod
    def _get_label_file(
        file_name: str, file_paths: List[str], extension: str, folder_name: str = ""
    ) -> str:
        """
        label file 구하는 함수
        """
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
        onsets_arr: dict, frame_length: int, hop_length: int
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
