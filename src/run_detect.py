import librosa
from model.separate_detect import SeparateDetectModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from constant import (
    ROOT_PATH,
    RAW_PATH,
    IDMT,
    ENST,
    E_GMD,
    SAMPLE_RATE,
    DDM_OWN,
    DRUM_KIT,
    METHOD_CLASSIFY,
    MFCC,
    PKL,
    METHOD_DETECT,
    MEL_SPECTROGRAM,
    CSV,
    METHOD_RHYTHM,
)

# predict_test_data = (
#     f"../data/raw/{ENST}/drummer_1/audio/hi-hat/003_hits_medium-tom_sticks_x5.wav"
# )
# predict_test_data = f"../data/raw/{DDM_OWN}/pattern/P2/16/P2_16_0001.m4a"
feature_path = f"{ROOT_PATH}/{RAW_PATH}"
predict_test_data = f"../data/raw/IDMT-SMT-DRUMS-V2/audio/RealDrum01_01#MIX.wav"
# predict_test_data = f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}/per-drum/HH/16/HH_16_0001.m4a"
# predict_test_data = (
#     "../data/new/e-gmd-v1.0.0/drummer6/session3/5_rock_180_beat_4-4_44.wav"
# )
separate_detect = SeparateDetectModel()
# separate_detect.extract_feature(feature_path)
separate_detect.run()
print(separate_detect.predict(predict_test_data, 100, 0))
