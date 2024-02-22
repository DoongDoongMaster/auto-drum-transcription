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
data_paths = [
    f"../data/raw/{DDM_OWN}/per-drum/HH/08/HH_08_0002.m4a",
    f"../data/raw/{DDM_OWN}/per-drum/KK/08/KK_08_0002.m4a",
    "../data/raw/drum-kit-sound/overheads/Overhead Sample 1.wav",
    "../data/raw/e-gmd-v1.0.0/drummer1/session2/5_jazz_200_beat_3-4.wav",
    # "../data/raw/e-gmd-v1.0.0/drummer1/session2/94_funk-rock_92_fill_4-4.wav",
    "../data/raw/ENST-drums-public/drummer_1/audio/hi-hat/001_hits_snare-drum_sticks_x6.wav",
    "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_01#MIX.wav",
    "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_57#KD.wav",
    f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}/per-drum/CC/04/CC_04_9949.m4a",
]
# FeatureExtractor.feature_extractor(data_paths, METHOD_DETECT, MEL_SPECTROGRAM, PKL)
# FeatureExtractor.load_feature_file(METHOD_DETECT, MEL_SPECTROGRAM, PKL)
predict_test_data = f"../data/raw/IDMT-SMT-DRUMS-V2/audio/RealDrum01_01#MIX.wav"
# predict_test_data = f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}/per-drum/HH/16/HH_16_0001.m4a"
# predict_test_data = (
#     "../data/new/e-gmd-v1.0.0/drummer6/session3/5_rock_180_beat_4-4_44.wav"
# )
separate_detect = SeparateDetectModel()
# separate_detect.run()
print(separate_detect.predict(predict_test_data, 100, 0))
