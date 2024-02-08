import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from constant import (
    SAMPLE_RATE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    CHUNK_LENGTH,
    FEATURE_PARAM,
    IMAGE_PATH,
    CLASSIFY_DURATION,
)


class AudioToFeature:
    """
    데이터에서 feature를 추출하는 클래스

    feature type: MFCC, STFT, MEL-SPECTOGRAM
    """

    @staticmethod
    def extract_feature(
        audio: np.ndarray, method_type: str, feature_type: str
    ) -> np.ndarray:
        """
        -- feature type에 따라 feature 추출
        """
        feature_param = FEATURE_PARAM[method_type][feature_type]
        frame_length = (CHUNK_LENGTH * SAMPLE_RATE) // feature_param["hop_length"]
        if method_type == METHOD_CLASSIFY:
            frame_length = (
                int(CLASSIFY_DURATION * SAMPLE_RATE) // feature_param["hop_length"]
            )

        feature_extraction_functions = {
            MFCC: AudioToFeature._audio_to_mfcc,
            STFT: AudioToFeature._audio_to_stft,
            MEL_SPECTROGRAM: AudioToFeature._audio_to_mel_spectrogram,
        }

        if feature_type not in feature_extraction_functions:
            raise ValueError("Invalid feature_type")

        result = feature_extraction_functions[feature_type](audio, feature_param)

        # classify 방식에서만 pad 채우기
        if method_type == METHOD_CLASSIFY:
            result = AudioToFeature._pad_feature(result, frame_length)

        if method_type in [METHOD_DETECT, METHOD_RHYTHM]:  # separate & detect방식 확인
            result = np.transpose(result)  # row: time, col: feature

        AudioToFeature._print_feature_info(audio, feature_type, result)
        return result

    @staticmethod
    def show_feature_plot(feature: np.ndarray, method_type: str, feature_type: str):
        """
        -- feature 그래프
        """
        # graph로 나타내기 위해 다시 transpose, row: feature, col: time
        if method_type in [METHOD_DETECT, METHOD_RHYTHM]:
            feature = np.transpose(feature)

        fig, ax = plt.subplots()
        db = (
            librosa.amplitude_to_db(feature, ref=np.max)
            if feature_type == STFT
            else librosa.power_to_db(feature, ref=np.max)
        )
        y_axis_info = "log" if feature_type == STFT else "mel"
        img = librosa.display.specshow(
            db,
            x_axis="time",
            y_axis=y_axis_info,
            sr=SAMPLE_RATE,
            ax=ax,
            fmax=FEATURE_PARAM[method_type][feature_type].get("fmax"),
            hop_length=FEATURE_PARAM[method_type][feature_type].get("hop_length"),
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="feature spectrogram")

        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/feature-{date_time}.png")
        plt.show()

    @staticmethod
    def _pad_feature(feature: np.ndarray, frame_length: int) -> np.ndarray:
        """
        -- feature의 padding 맞추는 함수
        """
        pad_width = frame_length - feature.shape[1]
        if pad_width > 0:
            feature = np.pad(
                feature, pad_width=((0, 0), (0, pad_width)), mode="constant"
            )
        else:
            feature = feature[:, :frame_length]
        return feature

    @staticmethod
    def _audio_to_mfcc(audio: np.ndarray, feature_param: dict) -> np.ndarray:
        """
        -- mfcc feature 추출
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=feature_param["n_mfcc"],
            hop_length=feature_param["hop_length"],
        )
        return mfccs

    @staticmethod
    def _audio_to_stft(audio: np.ndarray, feature_param: dict) -> np.ndarray:
        """
        -- stft feature 추출
        """
        stft = librosa.stft(
            y=audio,
            n_fft=feature_param["n_fft"],
            hop_length=feature_param["hop_length"],
            win_length=feature_param["win_length"],
            window="hann",
        )
        stft = np.abs(stft)
        return stft

    @staticmethod
    def _audio_to_mel_spectrogram(audio: np.ndarray, feature_param: dict) -> np.ndarray:
        """
        -- mel-spectrogram feature 추출
        """
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=feature_param["n_fft"],
            hop_length=feature_param["hop_length"],
            win_length=feature_param["win_length"],
            window="hann",
            n_mels=feature_param["n_mels"],
            fmin=feature_param["fmin"],
            fmax=feature_param["fmax"],
        )
        return mel_spectrogram

    @staticmethod
    def _print_feature_info(audio, feature_type, result):
        print(
            "-- length:",
            audio.shape[0] / float(SAMPLE_RATE),
            "secs, ",
            f"{feature_type}:",
            result.shape,
        )
