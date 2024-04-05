import librosa
import numpy as np
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from data.onset_detection import OnsetDetect
from model.segment_classify import SegmentClassifyModel
from model.separate_detect import SeparateDetectModel
from model.rhythm_detect_model import RhythmDetectModel

from constant import (
    CSV,
    DETECT_TYPES,
    E_GMD,
    IDMT,
    LABEL_DDM,
    LABEL_REF,
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
    TEST,
    TRAIN,
    VALIDATION,
    ENST,
)
from model.separate_detect_ref import SeparateDetectRefModel
from model.separate_detect_b import SeparateDetectBModel

# audio_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}")
# FeatureExtractor.feature_extractor(audio_paths, METHOD_DETECT, MEL_SPECTROGRAM, PKL)
# data = FeatureExtractor.load_feature_file(METHOD_DETECT, MEL_SPECTROGRAM, PKL)
# DataLabeling.show_label_plot(data["label"][400000:401200])
# AudioToFeature.show_feature_plot(data[400000:401200], METHOD_DETECT, MEL_SPECTROGRAM)


# predict_test_data = "../data/raw/ENST-drums-public/drummer_1/audio/snare/001_hits_snare-drum_sticks_x6.wav"
# segment_classify = SegmentClassifyModel()
# segment_classify.create_dataset()
# segment_classify.create()
# segment_classify.train()
# segment_classify.evaluate()
# print(segment_classify.predict(predict_test_data, 100, 0))

# file_path = "../data/raw/ENST-drums-public/drummer_1/audio/snare/001_hits_snare-drum_sticks_x6.wav"
# file_path = "../data/raw/IDMT-SMT-DRUMS-V2/audio/RealDrum01_00#MIX.wav"
# audio, _ = librosa.load(file_path, sr=44100)
# # print(len(OnsetDetect.get_onsets_using_librosa(audio, 441)))
# print(DataLabeling.data_labeling(audio, file_path, METHOD_CLASSIFY))

# if not any(p in file_path for p in CLASSIFY_ALL):
#     print("no")
# else:
#     print("포함")
# rhythm_detect = RhythmDetectModel(40, 0.01, 32, 16)

# rhythm_detect.create_dataset()
# rhythm_detect.create()
# rhythm_detect.train()
# rhythm_detect.evaluate()
# rhythm_detect.save()

# print(rhythm_detect.predict("../data/test/test_shifting.wav", 100, 0))
# print(rhythm_detect.predict("../data/test/004_hits_low-tom_sticks_x5.wav", 100, 0))

# rhythm_detect = RhythmDetectModel(40, 0.01, 32, 16)


# audio_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}")
# FeatureExtractor.feature_extractor(audio_paths, METHOD_DETECT, MEL_SPECTROGRAM, PKL)

# separate_detect = SeparateDetectModel(40, 0.01, 32, 128)

# separate_detect.create_dataset()
# separate_detect.create()
# separate_detect.train()
# separate_detect.evaluate()
# separate_detect.save()

# predict_test_datas = [
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
# "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_4.wav",
# ]
# for predict_test_data in predict_test_datas:
#     print(separate_detect.predict(predict_test_data, 100, 0))


# separate_detect_ref = SeparateDetectRefModel(40, 0.001, 32, 128)

# # == split_data, label_type 매개변수 바꿔서 사용!
# split_data = {TRAIN: [IDMT]}
# label_type = LABEL_DDM

# separate_detect_ref.create_dataset(split_data, label_type, DETECT_TYPES)
# separate_detect_ref.create()
# separate_detect_ref.train()
# separate_detect_ref.evaluate()
# separate_detect_ref.save()

separate_detect_ref = SeparateDetectBModel(40, 0.01, 32, 128)

# == split_data, label_type 매개변수 바꿔서 사용!
split_data = {TRAIN: [ENST], TEST: [ENST]}
label_type = LABEL_DDM

separate_detect_ref.create_dataset(split_data, label_type, DETECT_TYPES)
separate_detect_ref.create()
separate_detect_ref.train()
separate_detect_ref.evaluate()
separate_detect_ref.save()

# predict_test_datas = [
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
# "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_4.wav",
# ]
# for predict_test_data in predict_test_datas:
#     print(separate_detect_ref.predict(predict_test_data, 100, 0))


# ---------------------------------------------------------------------------------------
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display

# # bpm1 = "1_rock_90"
# bpm1 = "2_rock_100"
# # bpm1 = "3_rock_100"
# # bpm1 = "4_rock_127"
# # bpm1 = "5_rock_180"

# chunk_samples = int(6 * SAMPLE_RATE)

# y, sr = librosa.load(
#     f"../data/raw/librosa-peak-pick-test/{bpm1}_beat_4-4.wav",
#     sr=SAMPLE_RATE,
#     res_type="kaiser_fast",
# )
# y = y[:chunk_samples]
# y_harm, y_perc = librosa.effects.hpss(y)
# y = y_perc

# fig, ax = plt.subplots(sharex=True)
# librosa.display.waveshow(y, sr=sr, ax=ax, color="blue")
# onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=441)
# peaks = librosa.util.peak_pick(
#     onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
# )
# times = librosa.times_like(onset_env, sr=sr, hop_length=441)
# ax.vlines(times[peaks], -1, 1, color="r", alpha=0.8)

# # onset_env = np.array(onset_env)
# # # plt.plot(onset_env)
# # plt.plot(peaks, onset_env[peaks], "x")
# # plt.title("Model label")
# # plt.show()
# plt.savefig(f"new-output.png")
# # plt.savefig(f"{bpm1}_beat_4-4-output.png")
# --------------------------------------------------
# def merge_columns(arr, col1, col2):
#     # merge col2 into col1
#     # -- 둘 중 하나라도 1이면 1
#     # -- else, 둘 중 하나라도 0.5이면 0.5
#     # -- else, 0
#     merged_column = np.zeros(arr.shape[0])
#     for i in range(arr.shape[0]):
#         if 1 in arr[i, [col1, col2]]:
#             merged_column[i] = 1
#         elif 0.5 in arr[i, [col1, col2]]:
#             merged_column[i] = 0.5
#         else:
#             merged_column[i] = 0

#     # merge한 배열 col1 자리에 끼워넣기
#     result = np.delete(arr, [col1, col2], axis=1)
#     result = np.insert(result, col1, merged_column, axis=1)

#     return result

# arr = np.array(
#     [
#         [0, 1, 1],
#         [0, 0.5, 1],
#         [0, 0, 1],
#         [0.5, 1, 1],
#         [0.5, 0.5, 1],
#         [0.5, 0, 1],
#         [1, 1, 1],
#         [1, 0.5, 1],
#         [1, 0, 1],
#     ]
# )
# print(merge_columns(arr, 0, 1))
