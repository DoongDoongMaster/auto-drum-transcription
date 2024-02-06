import librosa
from model.segment_classify import SegmentClassifyModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from data.data_labeling import DataLabeling
from constant import (
    ROOT_PATH,
    RAW_PATH,
    IDMT,
    ENST,
    E_GMD,
    SAMPLE_RATE,
    DDM_OWN,
    DRUM_KIT,
)

# midi_path = (
#     "../data/raw/e-gmd-v1.0.0/drummer1/session1/97_neworleans-funk_84_fill_4-4.mid"
# )
# print(OnsetDetect.get_onsets_instrument_from_mid(midi_path))

# midi_path = f"../data/raw/{IDMT}/annotation_xml/RealDrum01_00#MIX.xml"
# OnsetDetect.get_onsets_from_xml(midi_path)

path = f"../data/raw/{DRUM_KIT}/kick/Bass Sample 1.wav"
audio, _ = librosa.load(path, sr=SAMPLE_RATE)
print(DataLabeling.get_onsets_instrument_arr(audio, path))

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
# segment_classify = SegmentClassifyModel()
# # segment_classify.extract_feature(f"{ROOT_PATH}/{RAW_PATH}/{IDMT}")
# # segment_classify.run()
# print(segment_classify.predict(predict_test_data, 100, 0))
