import os
import tensorflow as tf
from redisai import Client

from constant import CHUNK_TIME_LENGTH, MEL_SPECTROGRAM, METHOD_CLASSIFY, METHOD_DETECT
from data.data_processing import DataProcessing
from data.onset_detection import OnsetDetect
from feature.audio_to_feature import AudioToFeature
from feature.feature_extractor import FeatureExtractor
from model.base_model import BaseModel
from model.segment_classify import SegmentClassifyModel

# redisai_client = Client(host='localhost', port=6379)

#====success code!!!!! h5 -> graphdef (frozen model)==============
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# import numpy as np

# frozen_output_path = '../models/detect/frozen/' # frozen 모델을 저장할 경로
# frozen_model = 'frozen_test' # frozen 모델 이름

# model = tf.keras.models.load_model('../models/detect_mel-spectrogram_2024-03-15_16-36-20-[all]-[4]-[1-1-0.5]-[crnn(lstm)-acc(0.96)].h5') # tf_saved_model load

# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)) 
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()
# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 60)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)
# print("-" * 60)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir=frozen_output_path,
#                   name=f'{frozen_model}.pb',
#                   as_text=False)

# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir=frozen_output_path,
#                   name=f'{frozen_model}.pbtxt',
#                   as_text=True)
#==================================================

#=========redisAI model store=================
# model = open("../models/detect/frozen/frozen_test.pb", 'rb').read()

# redisai_client.modelstore('ddm-detect', 'tf', 'gpu', model, inputs=['x'], outputs=['Identity'])
#==============================================

# ==========check inputs, outputs name===========
# import tensorflow as tf
# gf = tf.compat.v1.GraphDef()
# gf.ParseFromString(open('../models/detect/frozen/frozen_test.pb','rb').read())
 
# print([n.name + '=>' +  n.op for n in gf.node if n.op])
# ==============================================


#===========redisAI predict (classify)================
# import librosa
# import numpy as np

# model = 'ddm-classify'
# model_meta = redisai_client.modelget(f'{model}', meta_only=True)
# if model_meta:
#     wav_path = "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#MIX.wav"
#     audio = FeatureExtractor.load_audio(wav_path)
#     # -- trimmed audio
#     onsets_arr = OnsetDetect.get_onsets_using_librosa(audio)
#     trimmed_audios = DataProcessing.trim_audio_per_onset(audio, onsets_arr)

#     # -- trimmed feature
#     predict_data = []
#     for _, taudio in enumerate(trimmed_audios):
#         trimmed_feature = AudioToFeature.extract_feature(
#             taudio, METHOD_CLASSIFY, MEL_SPECTROGRAM
#         )
#         predict_data.append(trimmed_feature)
#     # -- reshape
#     predict_data = SegmentClassifyModel.x_data_transpose(predict_data)

#     redisai_client.tensorset(f'{model}:in', predict_data)
#     # 3. Predict
#     redisai_client.modelexecute(model, inputs=[f'{model}:in'], outputs=[f'{model}:out'])

#     # 4. Get result
#     out = redisai_client.tensorget(f'{model}:out')
#     np.set_printoptions(precision=2, suppress=True)
#     print(out)

#=====================redisAI predict (detect) =================
# import librosa
# import numpy as np
# from sklearn.preprocessing import StandardScaler


# model = 'ddm-detect'
# model_meta = redisai_client.modelget(f'{model}', meta_only=True)
# if model_meta:
#     wav_path = "../data/test/IDMT-SMT-DRUMS-V2/audio/WaveDrum02_60#MIX.wav"
#     # Implement model predict logic
#     audio = FeatureExtractor.load_audio(wav_path)

#     # -- cut delay
#     new_audio = DataProcessing.trim_audio_first_onset(audio, 0)
#     audio = new_audio

#     # ------------------- compare predict with true label --------------------------
#     audio_feature = np.zeros((0, 128))

#     # 12s chunk하면서 audio feature추출 후 이어붙이기 -> 함수로 뽑을 예정
#     audios = DataProcessing.cut_chunk_audio(audio)
#     for i, ao in enumerate(audios):
#         # audio to feature
#         feature = AudioToFeature.extract_feature(
#             ao, METHOD_DETECT, MEL_SPECTROGRAM
#         )
#         audio_feature = np.vstack([audio_feature, feature])

#     scaler = StandardScaler()
#     audio_feature = scaler.fit_transform(audio_feature)

#     # -- input (#, time, 128 feature)
#     audio_feature = BaseModel.split_x_data(audio_feature, CHUNK_TIME_LENGTH)
#     audio_feature = audio_feature.astype(np.float32)

#     redisai_client.tensorset(f'{model}:in', audio_feature)
#     # 3. Predict
#     redisai_client.modelexecute(model, inputs=[f'{model}:in'], outputs=[f'{model}:out'])

#     # 4. Get result
#     out = redisai_client.tensorget(f'{model}:out')
#     np.set_printoptions(precision=2, suppress=True)
#     out = out.reshape((-1, 4))
#     print(out)