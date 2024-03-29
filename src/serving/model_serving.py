import numpy as npvvv
import tensorflow as tf

from redisai import Client
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from constant import MODEL_DIR, MODEL_SAVED_H5, MODEL_SAVED_PB, REDIS_AI_HOST, REDIS_AI_PORT, SERVED_MODEL_DIR


class ModelServing:
    def __init__(self, method_type:str, model_name:str,) -> None:
        self.redisai_client = Client(host=REDIS_AI_HOST, port=REDIS_AI_PORT)
        self.method_type = method_type
        self.model_name = model_name

    @staticmethod
    def convert_model_to_frozen(method_type: str, model_name:str):
        frozen_output_path = f'../{SERVED_MODEL_DIR}/{method_type}/' # frozen 모델을 저장할 경로
        frozen_model = model_name # frozen 모델 이름

        model = tf.keras.models.load_model(f'../{MODEL_DIR}/{model_name}.{MODEL_SAVED_H5}') # tf_saved_model load

        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)) 
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=frozen_output_path,
                            name=f'{frozen_model}.pb',
                            as_text=False)

        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=frozen_output_path,
                            name=f'{frozen_model}.pbtxt',
                            as_text=True)
        
    def store_model_to_server(self):
        model = open(f"../{SERVED_MODEL_DIR}/{self.method_type}/{self.model_name}.{MODEL_SAVED_PB}", 'rb').read()
        self.redisai_client.modelstore(self.model_name, 'tf', 'gpu', model, inputs=['x'], outputs=['Identity'])


    def predict_model_from_server(self):
        pass