import librosa
from feature.audio_to_feature import AudioToFeature

from constant import (
    SAMPLE_RATE,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
)

audio, _ = librosa.load(
    "../data/raw/ddm-own/pattern/P1/08/P1_08_0001.m4a", sr=SAMPLE_RATE
)

AudioToFeature.extract_feature(audio, MFCC, METHOD_RHYTHM)
