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

# audio_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}")
# FeatureExtractor.feature_extractor(audio_paths, METHOD_RHYTHM, MEL_SPECTROGRAM, PKL)
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

rhythm_detect = RhythmDetectModel(40, 0.01, 32, 16)

rhythm_detect.create_dataset()
rhythm_detect.create()
rhythm_detect.train()
rhythm_detect.evaluate()
rhythm_detect.save()

print(rhythm_detect.predict("../data/test/test_shifting.wav", 100, 0))
