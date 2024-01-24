import librosa
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from feature.feature_extractor import FeatureExtractor

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
    PKL,
)

audio_path = "../data/raw/ddm-own/pattern/P1/08/P1_08_0001.m4a"
audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

feature = AudioToFeature.extract_feature(audio, METHOD_CLASSIFY, MEL_SPECTROGRAM)
AudioToFeature.show_feature_plot(feature, METHOD_CLASSIFY, MEL_SPECTROGRAM)

# param = FEATURE_PARAM[METHOD_DETECT][MFCC]
# frame_length = (CHUNK_LENGTH * SAMPLE_RATE) // param["hop_length"]

# label = DataLabeling.get_label_ddm_detect(
#     audio, audio_path, frame_length, param["hop_length"]
# )

# feature_ext = FeatureExtractor(ROOT_PATH, METHOD_DETECT, MFCC, PKL)
# feature_ext.show_detect_label_plot(label)
