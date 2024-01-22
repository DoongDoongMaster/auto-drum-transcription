import os
import shutil
import scipy.io.wavfile

import librosa
import numpy as np
from typing import List

from constant import SAMPLE_RATE, RAW_PATH, NEW_PATH, ONSET_DURATION, CHUNK_LENGTH
from data.onset_detection import OnsetDetect

"""
data를 불러오고 처리하고 저장하는 클래스
"""


class DataProcessing:
    def __init__(
        self,
        data_root_path: str,
        sample_rate=SAMPLE_RATE,
        onset_duration=ONSET_DURATION,
        chunk_length=CHUNK_LENGTH,
    ):
        self.raw_data_path = f"{data_root_path}/{RAW_PATH}"
        self.new_data_path = f"{data_root_path}/{NEW_PATH}"
        self.sample_rate = sample_rate
        self.onset_duration = onset_duration
        self.onset_detection = OnsetDetect(sample_rate)
        self.chunk_length = chunk_length

    # get data path
    def get_paths(self, root_path: str, extensions=["m4a", "mp3", "wav"]) -> List[str]:
        if os.path.isfile(root_path):  # 파일이라면 불러오기
            if extensions[0] == "*":  # 모든 파일
                return [root_path]
            if any(root_path.endswith(ext) for ext in extensions):
                return [root_path]
            else:
                return []

        folders = os.listdir(root_path)
        audio_paths = []

        for d in folders:
            new_root_path = os.path.join(root_path, d)
            audio_paths += self.get_paths(new_root_path, extensions)

        return audio_paths

    # load audio data : 오디오 형태로 불러오기
    def load_audio_data(self, root_path: str):
        print("-- ! audio data loading ... ! --")
        audio_paths = self.get_paths(root_path)
        audios = [librosa.load(p, sr=self.sample_rate)[0] for p in audio_paths]
        print("-- ! audio data loading done ! --")
        return audios

    # check new data exist -> return boolean
    def is_exist_new_data(self):
        new_data_paths = self.get_paths(self.new_data_path)
        return len(new_data_paths) > 0

    # move new data to raw data
    def move_new_to_raw(self):
        print("-- ! moving new data to raw data ! --")
        new_data_paths = self.get_paths(self.new_data_path, ["*"])

        for p in new_data_paths:
            file_path = p.replace(NEW_PATH, RAW_PATH)  # new path의 폴더 경로를 유지하면서 옮기기
            file_dir = os.path.dirname(file_path)
            if os.path.exists(file_dir) == False:  # 파일 없다면 새로 생성
                os.makedirs(file_dir)
            shutil.move(p, file_path)

        # folder remove & remake
        if not self.is_exist_new_data():
            shutil.rmtree(self.new_data_path)
            os.mkdir(self.new_data_path)

        print("-- ! move done ! --")

    # trim audio per onset -> list
    def trim_audio_per_onset(self, audio: np.ndarray, onsets: List[float] = None):
        onsets = (
            self.onset_detection.onset_detection(audio) if onsets == None else onsets
        )
        sr = self.sample_rate
        duration = self.onset_duration

        trimmed_audios = []
        for i in range(0, len(onsets)):
            start = (int)((onsets[i]) * sr)
            end = (int)((onsets[i] + duration) * sr)

            if i + 1 < len(onsets):
                end_by_onset = (int)(onsets[i + 1] * sr)
                end = min(end, end_by_onset)

            trimmed = audio[start:end]
            trimmed_audios.append(trimmed)

        return trimmed_audios

    # trim audio from first onset to last audio
    def trim_audio_first_onset(self, audio: np.ndarray, first_onset: float = None):
        if first_onset == None:
            onsets = self.onset_detection.onset_detection(audio)
            first_onset = onsets[0] if len(onsets) != 0 else 0

        sr = self.sample_rate
        start = (int)(first_onset * sr)
        trimmed = audio[start:]

        print(f"-- ! audio trimmed: {first_onset} sec ! --")
        return trimmed

    def cut_chunk_audio(self, audio: np.ndarray):
        """
        chunk time씩 잘라 list로 리턴
        """
        # Calculate the number of samples in each chunk
        chunk_samples = int(self.chunk_length * self.sample_rate)

        # -- Cut audio into chunks
        audio_chunks = [
            audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)
        ]

        # Check if the last chunk is less than 1 second, discard if true
        if len(audio_chunks[-1]) < self.sample_rate:
            audio_chunks = audio_chunks[:-1]

        return audio_chunks

    # write wav audio -> wav file
    def write_wav_audio_one(self, root_path, name, audio):
        # exist or not
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        scipy.io.wavfile.write(f"{root_path}/{name}.wav", self.sample_rate, audio)

    # trimmed audio -> wav file write list
    # -- file name : 본래wavfile이름_몇번째onset인지.wav
    def write_trimmed_audio(self, root_path, name, trimmed_audios: List[np.ndarray]):
        start = 1
        for audio in trimmed_audios:
            self.write_wav_audio_one(root_path, f"{name}_{start:04}", audio)
            start += 1
