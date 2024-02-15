import librosa
from model.segment_classify import SegmentClassifyModel
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

# midi_path = (
#     "../data/raw/e-gmd-v1.0.0/drummer1/session1/97_neworleans-funk_84_fill_4-4.mid"
# )
# print(OnsetDetect.get_onsets_instrument_from_mid(midi_path))

# midi_path = f"../data/raw/{IDMT}/annotation_xml/RealDrum01_00#MIX.xml"
# OnsetDetect.get_onsets_from_xml(midi_path)

# path = f"../data/raw/{DRUM_KIT}/kick/Bass Sample 1.wav"
# audio, _ = librosa.load(path, sr=SAMPLE_RATE)
# OnsetDetect.get_peak_using_librosa(audio, 441)
# print(DataLabeling.get_onsets_instrument_arr(audio, path))

# midi_path = "../data/raw/drum-kit-sound/kick/Bass Sample 1.wav"
# # midi_path = "../data/raw/drum-kit-sound/snare/Snare Sample 10.wav"
# # midi_path = "../data/raw/ddm-own/pattern/P1/08/P1_08_0014.m4a"
# # audio, _ = librosa.load(midi_path, sr=SAMPLE_RATE)
# midi_paths = DataProcessing.get_paths(f"../data/raw/{DRUM_KIT}/kick/Bass Sample 1.wav")
# for path in midi_paths:
#     print(path)
#     audio, _ = librosa.load(path, sr=None)
#     # audio = audio[: 5 * SAMPLE_RATE]
#     onsets = OnsetDetect.get_onsets_using_librosa(audio)
#     audio = DataProcessing.trim_audio_first_onset(audio, onsets[0])
#     DataProcessing.write_wav_audio_one("../data/test", "test_drum_kit", audio)
# OnsetDetect.onset_detection(audio)

# onsets = OnsetDetect.get_onsets_from_svl(
#     "../data/raw/IDMT-SMT-DRUMS-V2/annotation_svl/WaveDrum02_01#HH.svl"
# )

# audio, _ = librosa.load(
#     f"../data/raw/{E_GMD}/drummer1/session1/1_funk_80_beat_4-4.wav", sr=44100
# )
# audio = audio[: 5 * 44100]
# onsets = OnsetDetect.get_onsets_using_librosa(audio, 441)
# trimmed = DataProcessing.trim_audio_per_onset(audio, onsets)
# DataProcessing.write_trimmed_audio(f"{ROOT_PATH}/test", "test", trimmed)

# predict_test_data = (
#     f"../data/raw/{ENST}/drummer_1/audio/hi-hat/003_hits_medium-tom_sticks_x5.wav"
# )
# predict_test_data = f"../data/raw/{DDM_OWN}/pattern/P2/16/P2_16_0001.m4a"
feature_path = f"{ROOT_PATH}/{RAW_PATH}"
# # midi_path = "../data/raw/e-gmd-v1.0.0/drummer1/session2/8_jazz-march_176_beat_4-4.mid"
# # OnsetDetect.get_onsets_instrument_from_mid(midi_path, end=5)
# predict_test_data = f"../data/raw/IDMT-SMT-DRUMS-V2/audio/RealDrum01_01#MIX.wav"
# predict_test_data = f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}/per-drum/HH/16/HH_16_0001.m4a"
predict_test_data = (
    "../data/new/e-gmd-v1.0.0/drummer6/session3/5_rock_180_beat_4-4_44.wav"
)
segment_classify = SegmentClassifyModel()
# segment_classify.extract_feature(feature_path)
# segment_classify.run()
print(segment_classify.predict(predict_test_data, 100, 0))

# -------------------
# Feature Extract
# data_paths_ddm = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{DDM_OWN}")
# data_paths_kit = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{DRUM_KIT}")
# data_paths_egmd=[]
# for i in range(3, 11):
#     data_paths_temp = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}/drummer{i}")
#     data_paths_egmd = data_paths_egmd + data_paths_temp
# data_paths = data_paths_egmd
# data_paths = [
#     # f"../data/raw/{DDM_OWN}/per-drum/HH/08/HH_08_0002.m4a",
#     f"../data/raw/{DDM_OWN}/per-drum/KK/08/KK_08_0002.m4a",
#     # "../data/raw/drum-kit-sound/overheads/Overhead Sample 1.wav",
#     # "../data/raw/e-gmd-v1.0.0/drummer1/session2/5_jazz_200_beat_3-4.wav",
#     # "../data/raw/e-gmd-v1.0.0/drummer1/session2/94_funk-rock_92_fill_4-4.wav",
#     # "../data/raw/ENST-drums-public/drummer_1/audio/hi-hat/001_hits_snare-drum_sticks_x6.wav",
#     # "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_01#MIX.wav",
#     # "../data/raw/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_57#KD.wav",
# ]
# FeatureExtractor.feature_extractor(data_paths, METHOD_CLASSIFY, MFCC, PKL)
# FeatureExtractor.load_feature_file(METHOD_CLASSIFY, MFCC, PKL)

# data = FeatureExtractor._load_feature_one_file(
#     "../data/processed-feature/classify/mfcc/mfcc-2024-02-08_11-15-29-0000.csv", CSV
# )
# AudioToFeature.show_feature_plot(data["feature"][1], METHOD_CLASSIFY, MFCC)
