import io
import logging

from fastapi import FastAPI, File, UploadFile

from constant import (
    SAMPLE_RATE,
    SERVED_MODEL_ALL,
)
from data.data_processing import DataProcessing
from feature.feature_extractor import FeatureExtractor
from serving.model_serving import ModelServing


app = FastAPI()
# 로그 생성
logger = logging.getLogger()
# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)


curr_model_num = 1
data = SERVED_MODEL_ALL[curr_model_num]
model_serving_class = ModelServing(
    data.get("method_type"),
    data.get("feature_type"),
    data.get("model_name"),
    data.get("label_cnt"),
)


@app.post("/adt/predict", tags=["ADT"])
def model_predict(file: bytes = File(...)):
    logger.info("I recived API server POST request")
    # ============ sercved model class create ========================

    logger.info("I'm waiting model prediction")
    # # Implement model predict logic
    bytesIO = io.BytesIO(file)
    audio = FeatureExtractor.load_audio(bytesIO)
    drum_instrument, onsets_arr = model_serving_class.predict_model_from_server(audio)
    logger.info("model prediction done !!!")

    # total wav file time (sec)
    audio_total_sec = len(audio) / SAMPLE_RATE

    return {
        "drum_instrument": drum_instrument,
        "onsets_arr": onsets_arr,
        "audio_total_sec": audio_total_sec,
    }
