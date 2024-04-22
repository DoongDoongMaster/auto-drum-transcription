import librosa
from model.segment_classify import SegmentClassifyModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from constant import (
    CLASSIFY_TYPES,
    ENST_PUB,
    MDB,
    NEW_PATH,
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
    TEST,
    TRAIN,
    VALIDATION,
)


# ============Feature Extract===================
# data_paths_ddm = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}")
# data_paths_kit = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{DRUM_KIT}")
# data_paths_enst = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{ENST}")
# data_paths_idmt = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{IDMT}")
# data_paths_egmd = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}")

# data_paths = data_paths_kit + data_path_enst + data_path_idmt
# data_paths_egmd=[]
# for i in range(3, 11):
#     data_paths_temp = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}/drummer{i}")
#     data_paths_egmd = data_paths_egmd + data_paths_temp
# data_paths = data_paths_egmd
# data_paths = [
#     f"../data/raw/{DDM_OWN}/per-drum/HH/08/HH_08_0002.m4a",
#     f"../data/raw/{DDM_OWN}/per-drum/KK/08/KK_08_0002.m4a",
#     "../data/raw/drum-kit-sound/overheads/Overhead Sample 1.wav",
#     "../data/raw/e-gmd-v1.0.0/drummer1/session2/5_jazz_200_beat_3-4.wav",
#     # "../data/raw/e-gmd-v1.0.0/drummer1/session2/94_funk-rock_92_fill_4-4.wav",
#     "../data/raw/ENST-drums-public/drummer_1/audio/hi-hat/001_hits_snare-drum_sticks_x6.wav",
#     "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_01#MIX.wav",
#     "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_57#KD.wav",
#     f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}/per-drum/CC/04/CC_04_9949.m4a",
# ]
# data_paths = [
#     "../data/raw/e-gmd-v1.0.0/drummer1/session1/78_jazz-fast_290_beat_4-4.wav"
# ]

# data_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{ENST}")
# FeatureExtractor.feature_extractor(data_paths, METHOD_DETECT, MEL_SPECTROGRAM, PKL)
# FeatureExtractor.load_feature_file(METHOD_DETECT, MEL_SPECTROGRAM, PKL, ENST, TRAIN)

# print(FeatureExtractor._load_feature_one_file(
#     "../data/processed-feature/classify/mel-spectrogram/mel-spectrogram-2024-03-13_01-26-48-0100.pkl", PKL
# ))
# AudioToFeature.show_feature_plot(data["feature"][1], METHOD_CLASSIFY, MFCC)

# ===============Model Train========================
# == split_data, label_type 매개변수 바꿔서 사용!
split_data = {
    TRAIN: [MDB, IDMT, ENST, E_GMD],
    # VALIDATION: [MDB, IDMT, ENST, E_GMD],
    TEST: [MDB, IDMT, ENST, E_GMD],
}

segment_classify = SegmentClassifyModel(
    training_epochs=50,
    batch_size=8,
    opt_learning_rate=0.001,
    feature_type=MEL_SPECTROGRAM,
)
# segment_classify.create_dataset(split_data, group_dict=CLASSIFY_TYPES)
# segment_classify.create()
# segment_classify.train()
# segment_classify.evaluate()
# segment_classify.save()

# ===============Model Evaluate========================
# == split_data, label_type 매개변수 바꿔서 사용!
# split_data = {
#     TEST: [
#         ENST,
#     ]
# }

# segment_classify = SegmentClassifyModel(
#     training_epochs=50,
#     batch_size=8,
#     opt_learning_rate=0.001,
#     feature_type=MEL_SPECTROGRAM,
# )
# segment_classify.create_dataset(split_data, group_dict=CLASSIFY_TYPES)
# segment_classify.evaluate()

# ===============Model Predict==========================
# segment_classify = SegmentClassifyModel(feature_type=MEL_SPECTROGRAM)

predict_test_datas = [
    "../data/test/e-gmd-v1.0.0/drummer1/session1/1_funk-groove1_138_beat_4-4.wav",
    "../data/test/e-gmd-v1.0.0/drummer1/session1/1_rock_105_beat_4-4.wav",
    "../data/test/e-gmd-v1.0.0/drummer1/session1/항해_솔로_일부분.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/162_MIDI-minus-one_fusion-125_sticks.wav",
    "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#HH.wav",
    "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#SD.wav",
    "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#KD.wav",
    "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#MIX.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/P1_08_0004.m4a",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/P2_16_0001.m4a",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_1.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_2.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_3.wav",
    "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_4.wav",
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/그로쓰_데모4.wav",
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/그로쓰_데모4_2.wav",
    # "../data/test/e-gmd-v1.0.0/drummer1/session1/데모용2.wav"
]
for predict_test_data in predict_test_datas:
    print(segment_classify.predict(predict_test_data, 90, 0))
