from constant import METHOD_CLASSIFY, METHOD_DETECT, SERVED_MODEL_CLASSIFY_BI_LSTM, SERVED_MODEL_CLASSIFY_LSTM, SERVED_MODEL_CLASSIFY_MFCC, SERVED_MODEl_DETECT_LSTM
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from serving.model_serving import ModelServing

# ============ sercved model class create ========================
model_serving_classify_lstm = ModelServing(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_LSTM)
model_serving_classify_bi_lstm = ModelServing(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_BI_LSTM)
model_serving_classify_mfcc = ModelServing(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_MFCC)
model_serving_detect_lstm = ModelServing(METHOD_DETECT, SERVED_MODEl_DETECT_LSTM)

# ============ model save ========================
# ModelServing.convert_model_to_frozen(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_LSTM)
# ModelServing.convert_model_to_frozen(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_BI_LSTM)
# ModelServing.convert_model_to_frozen(METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_MFCC)
# ModelServing.convert_model_to_frozen(METHOD_DETECT, SERVED_MODEl_DETECT_LSTM)

# model_serving_classify_lstm.store_model_to_server()
# model_serving_classify_bi_lstm.store_model_to_server()
# model_serving_classify_mfcc.store_model_to_server()
# model_serving_detect_lstm.store_model_to_server()

# ============= predict test =======================
wav_path = "../data/test/e-gmd-v1.0.0/drummer1/session1/voyage_solo.wav"
# # Implement model predict logic
audio = FeatureExtractor.load_audio(wav_path)
# -- cut delay
new_audio = DataProcessing.trim_audio_first_onset(audio, 0)
audio = new_audio
model_serving_detect_lstm.predict_model_from_server(audio)