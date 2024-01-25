import librosa
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from data.onset_detection import OnsetDetect

from constant import (
    SAMPLE_RATE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    FEATURE_PARAM,
    CHUNK_LENGTH,
    ROOT_PATH,
    RAW_PATH,
    NEW_PATH,
    PKL,
)

audio_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}")
# FeatureExtractor.feature_extractor(audio_paths, METHOD_RHYTHM, STFT, PKL)
data = FeatureExtractor.load_feature_file(METHOD_RHYTHM, STFT, PKL)
DataLabeling.show_label_plot(data["label"][:1200])
AudioToFeature.show_feature_plot(data[:1200], METHOD_RHYTHM, STFT)
