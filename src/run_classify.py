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
    CLASSIFY_ALL,
)

# audio_paths = DataProcessing.get_paths(f"{ROOT_PATH}/{RAW_PATH}")
# FeatureExtractor.feature_extractor(audio_paths, METHOD_CLASSIFY, MFCC, PKL)

predict_test_data = "../data/raw/IDMT-SMT-DRUMS-V2/audio/RealDrum01_00#KD#train.wav"
segment_classify = SegmentClassifyModel()
# segment_classify.create_dataset()
# segment_classify.create()
# segment_classify.train()
# segment_classify.evaluate()
# segment_classify.save()
print(segment_classify.predict(predict_test_data, 100, 0))
