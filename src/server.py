import base64
import io
from fastapi import FastAPI, File, UploadFile

from constant import (
    METHOD_CLASSIFY,
    METHOD_DETECT,
    SAMPLE_RATE,
    SERVED_MODEL_CLASSIFY_BI_LSTM,
    SERVED_MODEL_CLASSIFY_LSTM,
    SERVED_MODEL_CLASSIFY_MFCC,
    SERVED_MODEl_DETECT_LSTM,
)
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from serving.model_serving import ModelServing

app = FastAPI()


@app.post("/predict")
def model_predict(file: UploadFile = File(...)):
    # ============ sercved model class create ========================
    model_serving_classify_lstm = ModelServing(
        METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_LSTM
    )
    model_serving_classify_bi_lstm = ModelServing(
        METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_BI_LSTM
    )
    model_serving_classify_mfcc = ModelServing(
        METHOD_CLASSIFY, SERVED_MODEL_CLASSIFY_MFCC
    )
    model_serving_detect_lstm = ModelServing(METHOD_DETECT, SERVED_MODEl_DETECT_LSTM)

    # # Implement model predict logic
    audio = FeatureExtractor.load_audio(file.file)
    # -- cut delay
    new_audio = DataProcessing.trim_audio_first_onset(audio, 0)
    audio = new_audio
    drum_instrument, onsets_arr = model_serving_classify_lstm.predict_model_from_server(
        audio
    )

    audio_total_sec = len(audio) / SAMPLE_RATE

    return {
        "drum_instrument": drum_instrument,
        "onsets_arr": onsets_arr,
        "audio_total_sec": audio_total_sec,
    }
