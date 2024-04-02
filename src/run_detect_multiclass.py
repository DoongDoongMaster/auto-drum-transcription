import librosa
import numpy as np
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from data.onset_detection import OnsetDetect
from model.segment_classify import SegmentClassifyModel
from model.rhythm_detect_model import RhythmDetectModel

from constant import (
    CSV,
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
from model.separate_detect_multiclass import SeparateDetectMultiClassModel

separate_detect_multiclass = SeparateDetectMultiClassModel(40, 0.01, 32, 128)

# separate_detect_multiclass.create_dataset()
# separate_detect_multiclass.create()
# separate_detect_multiclass.train()
# separate_detect_multiclass.evaluate()
# separate_detect_multiclass.save()

predict_test_datas = [
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/1_funk-groove1_138_beat_4-4.wav",
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/1_rock_105_beat_4-4.wav",
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/항해_솔로_일부분.wav",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/162_MIDI-minus-one_fusion-125_sticks.wav",
    # "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#HH.wav",
    # "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#SD.wav",
    # "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#KD.wav",
    # "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#MIX.wav",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/P1_08_0004.m4a",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/P2_16_0001.m4a",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_1.wav",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_2.wav",
    # "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_3.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_4.wav",
    # "../data/test/record/dot.m4a",
    # "../data/test/record/vari.m4a",
]
for predict_test_data in predict_test_datas:
    print(separate_detect_multiclass.predict(predict_test_data, 100, 0))
