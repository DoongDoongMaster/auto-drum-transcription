from constant import (
    SERVED_MODEL_ALL,
)
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from serving.model_serving import ModelServing

# ============ sercved model class create & model save ========================
current_model_num = 0

model_serving_class = []
for data in SERVED_MODEL_ALL:
    temp_class = ModelServing(data.get("method_type"), data.get("feature_type"), data.get("model_name"), data.get("label_cnt"))
    model_serving_class.append(temp_class)
    if not data.get("is_frozen"):
        ModelServing.convert_model_to_frozen(data.get("method_type"), data.get("model_name"))
    if not data.get("is_stored"):
        temp_class.store_model_to_server()
    model_serving_class.append(temp_class)

# ============= predict test ====================================================
wav_path = "../data/test/ENST-drums-public-clean/drummer_1/audio/wet_mix/0329_demo_1.wav"
# # Implement model predict logic
audio = FeatureExtractor.load_audio(wav_path)
# -- cut delay
new_audio = DataProcessing.trim_audio_first_onset(audio, 0)
audio = new_audio

for c in model_serving_class:
    c.predict_model_from_server(audio)
