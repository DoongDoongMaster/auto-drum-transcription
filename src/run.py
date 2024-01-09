from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor

from constant import ROOT_PATH, PROCESSED_FEATURE, CLASSIFY, DETECT, MFCC, STFT


data_processing = DataProcessing(ROOT_PATH)

# lists = data_processing.get_paths(data_processing.new_data_path)
# print(lists)

# if data_processing.is_exist_new_data():
#     data_processing.move_new_to_raw()
#     lists = data_processing.get_paths(data_processing.new_data_path)
#     print(lists)

# train data
# audio_paths = data_processing.get_paths(data_processing.new_data_path)

# feature_extractor = FeatureExtractor(
#     data_root_path=ROOT_PATH,
#     middle_path=f"{PROCESSED_FEATURE}/{CLASSIFY}",
#     feature_type=MFCC,
#     n_classes=8,
#     n_features=40,
#     n_times=20,
#     n_channels=1,
# )

# feature_extractor.feature_extractor(audio_paths)

audio_paths = data_processing.get_paths(data_processing.new_data_path)

feature_extractor = FeatureExtractor(
    data_root_path=ROOT_PATH,
    middle_path=f"{PROCESSED_FEATURE}/{DETECT}",
    feature_type=STFT,
    n_features=40,
    n_times=1024,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
)

feature_extractor.feature_extractor(audio_paths)
