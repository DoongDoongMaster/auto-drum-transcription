import librosa
import numpy as np

from constant import (
    SAMPLE_RATE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    METHOD_DETECT,
    METHOD_RHYTHM,
    CHUNK_LENGTH,
    FEATURE_PARAM,
)


class AudioToFeature:
    """
    데이터에서 feature를 추출하는 클래스

    feature type: MFCC, STFT, MEL-SPECTOGRAM
    """

    @staticmethod
    def extract_feature(
        audio: np.ndarray, feature_type: str, method_type: str
    ) -> np.ndarray:
        """
        -- feature type에 따라 feature 추출
        """
        feature_param = FEATURE_PARAM[method_type][feature_type]

        method_requires_transpose = method_type in [
            METHOD_DETECT,
            METHOD_RHYTHM,
        ]  # separate & detect방식 확인

        frame_length = (CHUNK_LENGTH * SAMPLE_RATE) // feature_param["hop_length"]

        feature_extraction_functions = {
            MFCC: AudioToFeature._audio_to_mfcc,
            STFT: AudioToFeature._audio_to_stft,
            MEL_SPECTROGRAM: AudioToFeature._audio_to_mel_spectrogram,
        }

        if feature_type not in feature_extraction_functions:
            raise ValueError("Invalid feature_type")

        result = feature_extraction_functions[feature_type](
            audio, feature_param, frame_length
        )

        if method_requires_transpose:
            result = np.transpose(result)  # row: time, col: feature

        AudioToFeature._print_feature_info(audio, feature_type, result)
        return result

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
    def _audio_to_mfcc(
        audio: np.ndarray, feature_param: dict, frame_length: int
    ) -> np.ndarray:
        """
        -- mfcc feature 추출
        """
        mfccs = librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE, n_mfcc=feature_param["n_mfcc"]
        )
        mfccs = AudioToFeature._pad_feature(mfccs, frame_length)
        return mfccs

    @staticmethod
    def _audio_to_stft(
        audio: np.ndarray, feature_param: dict, frame_length: int
    ) -> np.ndarray:
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
        stft = np.abs(stft, dtype=np.float16)
        stft = AudioToFeature._pad_feature(stft, frame_length)
        return stft

    @staticmethod
    def _audio_to_mel_spectrogram(
        audio: np.ndarray, feature_param: dict, frame_length: int
    ) -> np.ndarray:
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
        mel_spectrogram = AudioToFeature._pad_feature(mel_spectrogram, frame_length)
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
