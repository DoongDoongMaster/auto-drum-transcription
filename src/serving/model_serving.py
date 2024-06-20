import os
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from redisai import Client
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from constant import (
    METHOD_CLASSIFY,
    METHOD_DETECT,
    MODEL_DIR,
    MODEL_SAVED_H5,
    MODEL_SAVED_PB,
    SERVED_MODEL_DETECT_EGMD_4,
    SERVED_MODEL_DIR,
)
from models.segment_classify import SegmentClassifyModel
from models.separate_detect import SeparateDetectModel


# .env 파일의 경로 설정
dotenv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../.env"
)
load_dotenv(dotenv_path)


class ModelServing:
    def __init__(
        self,
        method_type: str,
        feature_type: str,
        model_name: str,
        label_cnt: int = 4,
    ) -> None:
        self.redisai_client = Client(
            host=os.environ["REDIS_AI_HOST"], port=os.environ["REDIS_AI_PORT"]
        )
        self.method_type = method_type
        self.feature_type = feature_type
        self.model_name = model_name
        self.label_cnt = label_cnt

    @staticmethod
    def convert_model_to_frozen(method_type: str, model_name: str):
        frozen_output_path = (
            f"../{SERVED_MODEL_DIR}/{method_type}/"  # frozen 모델을 저장할 경로
        )
        frozen_model = model_name  # frozen 모델 이름

        model = tf.keras.models.load_model(
            f"../{MODEL_DIR}/{method_type}/{model_name}.{MODEL_SAVED_H5}"
        )  # tf_saved_model load

        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=frozen_output_path,
            name=f"{frozen_model}.pb",
            as_text=False,
        )

        # tf.io.write_graph(
        #     graph_or_graph_def=frozen_func.graph,
        #     logdir=frozen_output_path,
        #     name=f"{frozen_model}.pbtxt",
        #     as_text=True,
        # )

    def store_model_to_server(self):
        model = open(
            f"../{SERVED_MODEL_DIR}/{self.method_type}/{self.model_name}.{MODEL_SAVED_PB}",
            "rb",
        ).read()
        self.redisai_client.modelstore(
            self.model_name, "tf", "gpu", model, inputs=["x"], outputs=["Identity"]
        )

    def predict_model_from_server(self, audio: np.array):
        self.model_meta = self.redisai_client.modelget(
            f"{self.model_name}", meta_only=True
        )
        if self.model_meta:
            print("========== find model! ==============")
            model_class = None
            if self.method_type == METHOD_CLASSIFY:
                model_class = SegmentClassifyModel(
                    feature_type=self.feature_type, load_model_flag=False
                )
            elif self.method_type == METHOD_DETECT:
                model_class = SeparateDetectModel(load_model_flag=False)

            print("=========== model class ==========", model_class)
            print("=========== load model : ", self.model_name)

            # pre-processing
            input_data = model_class.data_pre_processing(audio)
            # data reshape for classify conv2d
            if self.method_type == METHOD_CLASSIFY and self.label_cnt < 5:
                input_data = model_class.input_reshape(input_data)
            if self.model_name == SERVED_MODEL_DETECT_EGMD_4:
                input_data = model_class.input_reshape(input_data)

            # make input data to redis ai
            self.redisai_client.tensorset(f"{self.model_name}:in", input_data)

            # predict
            self.redisai_client.modelexecute(
                self.model_name,
                inputs=[f"{self.model_name}:in"],
                outputs=[f"{self.model_name}:out"],
            )

            # get result from redis ai
            predict_result = self.redisai_client.tensorget(f"{self.model_name}:out")

            # post-processing
            drum_instrument, onsets_arr = model_class.data_post_processing(
                predict_result,
                audio,
                self.label_cnt,
            )

            # print("drum_instrument", drum_instrument)
            # print("onests_arr", onsets_arr)
            return drum_instrument, onsets_arr

        else:
            raise Exception("Not exists model")
