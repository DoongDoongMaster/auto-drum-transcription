import librosa
from model.segment_classify import SegmentClassifyModel
from data.onset_detection import OnsetDetect
from data.data_processing import DataProcessing
from constant import ROOT_PATH, RAW_PATH, IDMT, ENST, E_GMD

midi_path = "../data/raw/e-gmd-v1.0.0/drummer1/session1/1_funk_80_beat_4-4.mid"
OnsetDetect.get_onsets_instrument_midi(midi_path, 10, 20)

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
