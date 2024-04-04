from fastapi import FastAPI, File, UploadFile

from constant import (
    SAMPLE_RATE,
    SERVED_MODEL_ALL,
)
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from serving.model_serving import ModelServing

app = FastAPI()


@app.post("/predict")
def model_predict(file: UploadFile = File(...)):
    # ============ sercved model class create ========================
    curr_model_num = 5
    data = SERVED_MODEL_ALL[curr_model_num]
    model_serving_class = ModelServing(data.get("method_type"), data.get("feature_type"), data.get("model_name"), data.get("label_cnt"))

    # # Implement model predict logic
    audio = FeatureExtractor.load_audio(file.file)
    # -- cut delay
    new_audio = DataProcessing.trim_audio_first_onset(audio, 0)
    audio = new_audio
    drum_instrument, onsets_arr = model_serving_class.predict_model_from_server(
        audio
    )

    # total wav file time (sec)
    audio_total_sec = len(audio) / SAMPLE_RATE

    return {
        "drum_instrument": drum_instrument,
        "onsets_arr": onsets_arr,
        "audio_total_sec": audio_total_sec,
    }
