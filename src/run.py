import librosa
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from data.onset_detection import OnsetDetect
from model.segment_classify import SegmentClassifyModel
from model.rhythm_detect_model import RhythmDetectModel

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
FeatureExtractor.feature_extractor(audio_paths, METHOD_RHYTHM, MEL_SPECTROGRAM, PKL)
# data = FeatureExtractor.load_feature_file(METHOD_RHYTHM, MEL_SPECTROGRAM, PKL)
# DataLabeling.show_label_plot(data["label"][400000:401200])
# AudioToFeature.show_feature_plot(data[400000:401200], METHOD_RHYTHM, MEL_SPECTROGRAM)

# segment_classify = SegmentClassifyModel()
# # segment_classify.create_dataset()
# # segment_classify.create()
# # segment_classify.train()
# # segment_classify.evaluate()
# print(
#     segment_classify.predict(
#         "../data/raw/ENST-drums-public/drummer_1/audio/snare/001_hits_snare-drum_sticks_x6.wav",
#         100,
#         0,
#     )
# )

# rhythm_detect = RhythmDetectModel(40, 0.01, 32, 16)

# rhythm_detect.create_dataset()
# rhythm_detect.create()
# rhythm_detect.train()
# rhythm_detect.evaluate()
# rhythm_detect.save()

# print(rhythm_detect.predict("../data/test/test_shifting.wav", 100, 0))
# print(rhythm_detect.predict("../data/test/004_hits_low-tom_sticks_x5.wav", 100, 0))

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
