import librosa
from feature.audio_to_feature import AudioToFeature
from data.data_labeling import DataLabeling
from feature.feature_extractor import FeatureExtractor
from data.onset_detection import OnsetDetect

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
    PKL,
)

# audio_path = "../data/raw/ddm-own/pattern/P1/08/P1_08_0001.m4a"
# audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

# feature = AudioToFeature.extract_feature(audio, METHOD_CLASSIFY, MEL_SPECTROGRAM)
# AudioToFeature.show_feature_plot(feature, METHOD_CLASSIFY, MEL_SPECTROGRAM)

# # lists = data_processing.get_paths(data_processing.new_data_path)
# # print(lists)

# # if data_processing.is_exist_new_data():
# #     data_processing.move_new_to_raw()
# #     lists = data_processing.get_paths(data_processing.new_data_path)
# #     print(lists)

# # # train data
# # audio_paths = data_processing.get_paths(data_processing.new_data_path)

# feature_extractor = FeatureExtractor(
#     data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
#     method_type=METHOD_RHYTHM,
#     feature_type=MEL_SPECTROGRAM,
# )
# audio_paths = data_processing.get_paths(data_processing.raw_data_path)
# # feature_extractor.rhythm_feature_extractor(audio_paths)

# feature_extractor.feature_extractor(audio_paths)

# features = feature_extractor.load_feature_file()
# # print("feature><<", feature["label"])
# feature_extractor.show_rhythm_label_plot(features.label[0])

# mid_path = "../data/test/RealDrum01_00#MIX.xml"
# onsets_arr = OnsetDetect.get_onsets_from_xml(mid_path)
# chunk = feature_ext.split_onset_match_chunk(onsets_arr)

# onset2 = onsets_arr = OnsetDetect.get_onsets_from_xml(
#     mid_path, CHUNK_LENGTH, CHUNK_LENGTH * 2
# )
# print(chunk)
# print(onset2)

audio_path = "../data/raw/ddm-own/pattern/P1/08/P1_08_0001.m4a"
audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
label = DataLabeling.data_labeling(audio, audio_path, METHOD_CLASSIFY, 7, 1200, 441)
DataLabeling.show_label_plot(label)

# segment_classify = SegmentClassifyModel(40, 0.001, 20)

# segment_classify.create_dataset()
# segment_classify.create()
# segment_classify.train()
# segment_classify.evaluate()
# segment_classify.save()

# print(segment_classify.predict("../data/raw/pattern/P1/08/P1_08_0001.m4a", 100, 0))

# separate_detect = SeparateDetectModel(40, 0.001, 20, 16)

# separate_detect.create_dataset()
# separate_detect.create()
# separate_detect.train()
# separate_detect.evaluate()
# separate_detect.save()

# print(separate_detect.predict("../data/raw/pattern/P1/08/P1_08_0001.m4a", 100, 0))

rhythm_detect = RhythmDetectModel(40, 0.01, 32, 16)

rhythm_detect.create_dataset()
rhythm_detect.create()
rhythm_detect.train()
rhythm_detect.evaluate()
# rhythm_detect.save()

# print(rhythm_detect.predict("../data/test/test_shifting.wav", 100, 0))
