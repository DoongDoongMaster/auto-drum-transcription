import os
import shutil
import scipy.io.wavfile

import librosa
import numpy as np
from typing import List

from constant import (
    ONSET_DURATION_RIGHT_MINUS,
    SAMPLE_RATE,
    RAW_PATH,
    NEW_PATH,
    ONSET_DURATION_LEFT,
    ONSET_DURATION_RIGHT,
    CHUNK_LENGTH,
)
from data.onset_detection import OnsetDetect


class DataProcessing:
    """
    data를 불러오고 처리하고 저장하는 클래스
    """

    @staticmethod
    def get_paths(root_path: str, extensions=["m4a", "mp3", "wav"]) -> List[str]:
        """
        get data path
        """
        if os.path.isfile(root_path):  # 파일이라면 불러오기
            if extensions[0] == "*" or any(
                root_path.endswith(ext) for ext in extensions
            ):  # 모든 파일 or 특정 확장자
                return [root_path]
            else:
                return []

        audio_paths = []
        folders = os.listdir(root_path)

        for d in folders:
            new_root_path = os.path.join(root_path, d)
            audio_paths += DataProcessing.get_paths(new_root_path, extensions)

        return audio_paths

    @staticmethod
    def load_audio_data(root_path: str):
        """
        load audio data : 오디오 형태로 불러오기
        """
        print("-- ! audio data loading ... ! --")
        audio_paths = DataProcessing.get_paths(root_path)
        audios = [librosa.load(p, sr=SAMPLE_RATE)[0] for p in audio_paths]
        print("-- ! audio data loading done ! --")
        return audios

    @staticmethod
    def is_exist_data_in_folder(folder_path) -> bool:
        """
        check data exist
        """
        data_paths = DataProcessing.get_paths(folder_path)
        return len(data_paths) > 0

    @staticmethod
    def move_new_to_raw(new_path, raw_path):
        """
        move new data to raw data
        """
        print(f"-- ! moving {new_path} data to {raw_path} ! --")
        new_data_paths = DataProcessing.get_paths(new_path, ["*"])

        for p in new_data_paths:
            file_path = p.replace(
                NEW_PATH, RAW_PATH
            )  # new path의 폴더 경로를 유지하면서 옮기기
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)  # 파일 없다면 새로 생성
            shutil.move(p, file_path)

        print("-- ! move done ! --")

    @staticmethod
    def trim_audio_per_onset(
        audio: np.ndarray, onsets: List[float] = None
    ) -> List[np.ndarray]:
        """
        trim audio per onset
        """
        onsets = (
            OnsetDetect.get_onsets_using_librosa(audio) if onsets is None else onsets
        )

        trimmed_audios = []
        for i in range(0, len(onsets)):
            start = max(int((onsets[i] - ONSET_DURATION_LEFT) * SAMPLE_RATE), 0)
            end = int((onsets[i] + ONSET_DURATION_RIGHT) * SAMPLE_RATE)

            if i + 1 < len(onsets):
                end_by_onset = int((onsets[i + 1] - ONSET_DURATION_LEFT) * SAMPLE_RATE)
                end = min(end, end_by_onset)

            trimmed = audio[start:end]
            trimmed_audios.append(trimmed)

        return trimmed_audios

    @staticmethod
    def trim_audio_per_onset_with_duration(
        audio: np.ndarray, onsets: List[dict]
    ) -> List[np.ndarray]:
        """
        trim audio per onset (duration이 지정된 경우)
        """
        trimmed_audios = []
        for i in range(0, len(onsets)):
            start = max(
                int((onsets[i]["onset"] - ONSET_DURATION_LEFT) * SAMPLE_RATE), 0
            )
            end_duration = min(
                onsets[i]["duration"] - ONSET_DURATION_RIGHT_MINUS, ONSET_DURATION_RIGHT
            )
            end = int(float(onsets[i]["onset"] + end_duration) * SAMPLE_RATE)

            trimmed = audio[start:end]
            trimmed_audios.append(trimmed)

        return trimmed_audios

    @staticmethod
    def trim_audio_first_onset(audio: np.ndarray, first_onset: float = None):
        """
        trim audio from first onset to last audio
        """
        if first_onset == None:
            onsets = OnsetDetect.get_onsets_using_librosa(audio)
            first_onset = onsets[0] if len(onsets) > 0 else 0

        start = max(int((first_onset - ONSET_DURATION_LEFT) * SAMPLE_RATE), 0)
        trimmed = audio[start:]

        print(f"-- ! audio trimmed: {first_onset} sec ! --")
        return trimmed

    @staticmethod
    def cut_chunk_audio(audio: np.ndarray):
        """
        chunk time씩 잘라 list로 리턴
        """
        # Calculate the number of samples in each chunk
        chunk_samples = int(CHUNK_LENGTH * SAMPLE_RATE)

        # -- Cut audio into chunks
        audio_chunks = [
            audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)
        ]

        # Check if the last chunk is less than 1 second, discard if true
        if len(audio_chunks[-1]) < SAMPLE_RATE:
            audio_chunks = audio_chunks[:-1]

        return audio_chunks

    @staticmethod
    def write_wav_audio_one(root_path, name, audio):
        """
        write wav audio -> wav file
        """
        os.makedirs(root_path, exist_ok=True)
        scipy.io.wavfile.write(f"{root_path}/{name}.wav", SAMPLE_RATE, audio)

    @staticmethod
    def write_trimmed_audio(root_path, name, trimmed_audios: List[np.ndarray]):
        """
        trimmed audio -> wav file write list
        -- file name : 본래wavfile이름_몇번째onset인지.wav
        """
        for i, audio in enumerate(trimmed_audios, start=1):
            DataProcessing.write_wav_audio_one(root_path, f"/{name}_{i:04}", audio)

    @staticmethod
    def convert_array_dtype_float32(data):
        data_float32 = data.astype(np.float32)
        return data_float32
