from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from model.segment_classify import SegmentClassifyModel
from model.separate_detect import SeparateDetectModel
from model.rhythm_detect_model import RhythmDetectModel

from constant import (
    ROOT_PATH,
    PROCESSED_FEATURE,
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
    MFCC,
    STFT,
    MEL_SPECTROGRAM,
    FEATURE_PARAM,
)


# data_processing = DataProcessing(ROOT_PATH)

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

# audio_paths = data_processing.get_paths(data_processing.new_data_path)
# print(audio_paths)

# feature_dict = FEATURE_PARAM[METHOD_CLASSIFY][STFT]
# print(feature_dict)

# feature_extractor = FeatureExtractor(
#     data_root_path=f"{ROOT_PATH}/{PROCESSED_FEATURE}",
#     method_type=METHOD_CLASSIFY,
#     feature_type=STFT,
#     n_fft=feature_dict["n_fftt"],
#     n_times=feature_dict.n_times,
#     hop_length=feature_dict.hop_length,
#     win_length=feature_dict.win_length,
# )

# feature_extractor.feature_extractor(audio_paths)

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
rhythm_detect.save()

print(rhythm_detect.predict("../data/test/test_shifting.wav", 100, 0))
