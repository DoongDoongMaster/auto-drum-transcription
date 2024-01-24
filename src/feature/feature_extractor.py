import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import mido
import pretty_midi

from ast import literal_eval
from typing import List
from glob import glob
from datetime import datetime

from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect

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

"""
데이터에서 feature를 추출하고, 라벨링하고, 저장하는 클래스
"""


class FeatureExtractor:
    def __init__(
        self,
        data_root_path,
        method_type,
        feature_type,
        # feature_param: dict,
        feature_extension=PKL,
        chunk_length=CHUNK_LENGTH,
    ):
        self.data_root_path = data_root_path
        self.method_type = method_type
        self.feature_type = feature_type
        self.sample_rate = SAMPLE_RATE
        self.feature_param = FEATURE_PARAM[method_type][feature_type]
        self.frame_length = (CHUNK_LENGTH * SAMPLE_RATE) // self.feature_param[
            "hop_length"
        ]
        self.feature_extension = feature_extension
        self.save_path = (
            f"{data_root_path}/{method_type}/{feature_type}.{feature_extension}"
        )
        self.onset_detection = OnsetDetect(SAMPLE_RATE)
        self.data_processing = DataProcessing(data_root_path=ROOT_PATH)
        self.chunk_length = chunk_length

    """
    -- feature 추출한 파일 불러오기
    """

    def load_feature_file(self):
        # data_feature_label = None

        # combined_df = pd.DataFrame(columns=["feature", "label"])
        combined_df = pd.DataFrame(
            columns=["label"] + ["mel-spec" + str(i + 1) for i in range(128)]
        )

        save_folder_path = (
            f"{ROOT_PATH}/{PROCESSED_FEATURE}/{METHOD_RHYTHM}/{MEL_SPECTROGRAM}/"
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
    -- mfcc feature 추출
    """

    def audio_to_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.feature_param["n_mfcc"]
        )
        pad_width = self.frame_length - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfccs = mfccs[:, : self.frame_length]
        return mfccs

    """
    -- stft feature 추출
    """

    def audio_to_stft(self, audio: np.ndarray) -> np.ndarray:
        # translate STFT
        stft = librosa.stft(
            y=audio,
            n_fft=self.feature_param["n_fft"],
            hop_length=self.feature_param["hop_length"],
            win_length=self.feature_param["win_length"],
            window="hann",
        )
        stft = np.abs(stft, dtype=np.float64)
        if stft.shape[1] < self.frame_length:
            stft_new = np.pad(
                stft,
                pad_width=((0, 0), (0, self.frame_length - stft.shape[1])),
                mode="constant",
            )
        else:
            stft_new = stft[:, : self.frame_length]

        return stft_new

    """
    -- mel-spectrogram feature 추출
    """

    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        # translate mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.feature_param["n_fft"],
            hop_length=self.feature_param["hop_length"],
            win_length=self.feature_param["win_length"],
            window="hann",
            n_mels=self.feature_param["n_mels"],
            fmin=self.feature_param["fmin"],
            fmax=self.feature_param["fmax"],
        )
        # show graph
        # self.show_mel_spectrogram_plot(mel_spectrogram)

        if mel_spectrogram.shape[1] < self.frame_length:
            mel_spectrogram_new = np.pad(
                mel_spectrogram,
                pad_width=(
                    (0, 0),
                    (0, self.frame_length - mel_spectrogram.shape[1]),
                ),
                mode="constant",
            )
        else:
            mel_spectrogram_new = mel_spectrogram[:, : self.frame_length]

        return mel_spectrogram_new

    """
    -- feature type에 따라 feature 추출
    """

    def audio_to_feature(self, audio: np.ndarray) -> np.ndarray:
        result = None
        if self.feature_type == MFCC:
            result = self.audio_to_mfcc(audio)
        elif self.feature_type == STFT:
            result = self.audio_to_stft(audio)
        elif self.feature_type == MEL_SPECTROGRAM:
            result = self.audio_to_mel_spectrogram(audio)

        # if (
        #     self.method_type == METHOD_DETECT or self.method_type == METHOD_RHYTHM
        # ):  # separate & detect방식이라면 transpose
        #     result = np.transpose(result)  # row: time, col: feature

        print(
            "-- length:",
            audio.shape[0] / float(self.sample_rate),
            "secs, ",
            f"{self.feature_type}:",
            result.shape,
        )
        return result

    """
    -- 우리 데이터 기준 classify type (trimmed data) 라벨링
    """

    def get_label_classify_data(self, idx: int, path: str) -> List[int]:
        file_name = os.path.basename(path)  # extract file name
        if PATTERN_DIR in path:  # -- pattern
            pattern_name = file_name[:2]  # -- P1
            label = PATTERN2CODE[pattern_name][idx]
        elif PER_DRUM_DIR in path:  # -- per drum
            drum_name = file_name[:2]  # -- CC
            label = ONEHOT_DRUM2CODE[drum_name]
        return label

    """
    -- 우리 데이터 기준 detect type (sequence data) 라벨링 
        onset position : 1
        onset position with ONSET_OFFSET : 0.5 (ONSET_OFFSET: onset position 양 옆으로 몇 개씩 붙일지)
        extra : 0
    """

    def get_audio_position(self, time) -> int:
        return int(time * self.sample_rate / self.feature_param["hop_length"])

    def get_label_detect_data(self, audio: np.ndarray, path: str) -> List[List[int]]:
        file_name = os.path.basename(path)
        onsets_arr = self.onset_detection.onset_detection(audio)
        last_time = (
            self.frame_length * self.feature_param["hop_length"] / self.sample_rate
        )

        labels = [[0.0] * len(CODE2DRUM) for _ in range(self.frame_length)]
        pattern_idx = 0
        for onset in onsets_arr:
            if onset >= last_time:
                break

            onset_position = self.get_audio_position(onset)
            soft_start_position = max(  # -- onset - offset
                (onset_position - ONSET_OFFSET), 0
            )
            soft_end_position = min(  # -- onset + offset
                onset_position + ONSET_OFFSET, self.get_audio_position(last_time)
            )

            if any(drum in file_name for _, drum in CODE2DRUM.items()):  # per drum
                one_hot_label: List[int] = ONEHOT_DRUM2CODE[file_name[:2]]
            else:  # pattern
                pattern_name = file_name[:2]  # -- P1
                one_hot_label: List[int] = PATTERN2CODE[pattern_name][pattern_idx]
                pattern_idx += 1
            for i in range(soft_start_position, soft_end_position):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = (np.array(one_hot_label) / 2).tolist()
            labels[(int)(onset_position)] = one_hot_label

        return labels

    """
    -- XML file 읽기
    """

    def load_xml_file(self, file_path):
        try:
            # XML 파일을 파싱합니다.
            tree = ET.parse(file_path)
            # 루트 엘리먼트를 얻습니다.
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            print(f"XML 파일을 파싱하는 동안 오류가 발생했습니다: {e}")
            return None

    """
    -- XML file에서 onset 읽어오기
    """

    def get_onsets_arr_from_xml(self, xml_path: str):
        print("-- ! xml file location: ", xml_path)

        onset_sec_list = []
        xml_root = self.load_xml_file(xml_path)
        # transcription 엘리먼트의 정보 출력
        transcription_element = xml_root.find(".//transcription")
        events = transcription_element.findall("event")
        for event in events:
            onset_sec = event.find("onsetSec").text
            onset_sec_list.append(float(onset_sec))

        print("-- ! 파싱한 onsets: ", onset_sec_list)
        return onset_sec_list

    """
    -- TXT file에서 onset 읽어오기
    """

    def get_onsets_arr_from_txt(self, txt_path: str):
        print("-- ! txt file location: ", txt_path)

        onset_sec_list = []
        with open(txt_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                # 각 라인에서 onset 정보를 추출하여 리스트에 추가
                onset_sec = line.split()[0]
                onset_sec_list.append(float(onset_sec))

        print("-- ! 읽은 onsets: ", onset_sec_list)
        return onset_sec_list

    """
    -- MID file에서 onset 읽어오기
    """

    def get_onsets_arr_from_mid(self, midi_path: str):
        # MIDI 파일을 PrettyMIDI 객체로 로드합니다.
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # onset 정보를 저장할 리스트를 생성합니다.
        onset_times = []

        # 각 악기(track)에 대해 처리합니다.
        for instrument in midi_data.instruments:
            # 악기의 노트를 순회하며 onset을 찾습니다.
            for note in instrument.notes:
                onset_times.append(note.start)

        # onset_times를 정렬합니다.
        onset_times.sort()

        return onset_times

    """
    -- onset 라벨링 (ONSET_OFFSET: onset position 양 옆으로 몇 개씩 붙일지)
    """

    def get_label_rhythm_data(self, last_time, onsets_arr: List[float]) -> List[float]:
        labels = [0.0] * self.frame_length

        for onset in onsets_arr:
            onset_position = self.get_audio_position(onset)  # -- 1

            if onset_position >= self.frame_length:
                break

            soft_start_position = max(  # -- onset - offset
                (onset_position - ONSET_OFFSET), 0
            )
            soft_end_position = min(  # -- onset + offset
                onset_position + ONSET_OFFSET + 1, self.get_audio_position(last_time)
            )
            # offset -> 양 옆으로 0.5 몇 개 붙일지
            for i in range(soft_start_position, soft_end_position):
                if i >= self.frame_length:
                    break

                if labels[i] == 1.0:
                    continue
                labels[i] = 0.5

            labels[onset_position] = 1.0

        return labels

    """
    -- onset을 chunk에 맞게 split
    ex. {0: [0~11], 1: [12, 23], 2: [], 3: [38], ... onset 을 12배수에 따라 split
    -> 0~11
    """

    def split_onset_match_chunk(self, onsets_arr: List[float]):
        chunk_onsets_arr = {}
        tmp = []
        current_chunk_idx = 0
        for onset_time in onsets_arr:
            if (
                current_chunk_idx * self.chunk_length <= onset_time
                and onset_time < (current_chunk_idx + 1) * self.chunk_length
            ):
                tmp.append(
                    onset_time - (current_chunk_idx * self.chunk_length)
                    if onset_time >= self.chunk_length
                    else onset_time
                )
                continue
            chunk_onsets_arr[current_chunk_idx] = tmp
            current_chunk_idx += 1
            tmp = []

        chunk_onsets_arr[current_chunk_idx] = tmp
        return chunk_onsets_arr

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
            audio, _ = librosa.load(path, sr=self.sample_rate, res_type="kaiser_fast")

            if DDM_OWN in path:  # 우리 데이터라면
                # -- trimmed audio
                trimmed_audios = self.data_processing.trim_audio_per_onset(audio)
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
            audio, _ = librosa.load(path, sr=self.sample_rate, res_type="kaiser_fast")

            if DDM_OWN in path:  # 우리 데이터라면
                # -- trim first onset
                audio = self.data_processing.trim_audio_first_onset(audio)
                # -- feature extract
                feature = self.audio_to_feature(audio)
                # -- label: 드럼 종류마다 onset 여부
                label = self.get_label_detect_data(audio, path)
                data_feature_label.append([feature.tolist(), label])

        feature_df = pd.DataFrame(data_feature_label, columns=["feature", "label"])
        return feature_df

    """
    -- detect type label 그래프
    """

    def show_detect_label_plot(self, label: List[List[float]]):
        label = np.array(label)
        for i in range(len(CODE2DRUM)):
            plt.subplot(8, 1, i + 1)
            plt.plot(label[:, i])
        plt.title("Model label")
        plt.show()

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
            audio, _ = librosa.load(path, sr=self.sample_rate, res_type="kaiser_fast")

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
            chunk_list = self.data_processing.cut_chunk_audio(audio)
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
                    len(chunk) / self.sample_rate, chunk_onsets_arr[idx]
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
    -- rhythm type label 그래프
    """

    def show_rhythm_label_plot(self, label: List[float]):
        label = np.array(label)
        plt.plot(label)
        plt.title("Model label")
        plt.show()

        # 이미지 폴더 존재 확인
        if not os.path.exists(IMAGE_PATH):
            os.mkdir(IMAGE_PATH)  # 없으면 새로 생성

        # 현재 날짜와 시간 가져오기
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{IMAGE_PATH}/{self.method_type}-label-test-{date_time}.png")

    """
    -- mel-spectrogram 그래프
    """

    def show_mel_spectrogram_plot(self, mel_spectrogram: np.ndarray):
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
        img = librosa.display.specshow(
            S_dB,
            x_axis="time",
            y_axis="mel",
            sr=self.sample_rate,
            ax=ax,
            fmax=self.feature_param["fmax"],
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Mel-frequency spectrogram")
        # plt.savefig("mel-spectrogram-test.png")

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
