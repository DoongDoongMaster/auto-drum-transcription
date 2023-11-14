import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

# 데이터 전처리 함수
def preprocess_data(wav_file_path, labels):
    # WAV 파일을 스펙트로그램으로 변환하고 필요한 전처리 수행
    # ...

    # 라벨링된 데이터를 이진 벡터와 시작 시간으로 분리
    binary_vectors = labels[:, :8]
    start_times = labels[:, 8]
    return spectrogram_data, binary_vectors, start_times

# 데이터 불러오기 및 전처리
wav_file_path = "path_to_your_wav_file.wav"
labels = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],  # 예시 라벨링 데이터
                   [0, 0, 0, 0, 1, 0, 0, 0, 100]])  # 예시 라벨링 데이터
input_data, binary_vectors, start_times = preprocess_data(wav_file_path, labels)

# 모델 입력의 형태 설정
input_shape = input_data.shape[1:]

# 이진 벡터 예측 레이어
binary_vector_output = Dense(8, activation='sigmoid', name='binary_vector')(x)

# 시작 시간 예측 레이어
start_time_output = Dense(1, name='start_time')(x)

# 모델 정의
model = Model(inputs=input_layer, outputs=[binary_vector_output, start_time_output])

# 모델 컴파일
model.compile(optimizer='adam', loss={'binary_vector': 'binary_crossentropy', 'start_time': 'mean_squared_error'})

# 모델 훈련
model.fit(input_data, {'binary_vector': binary_vectors, 'start_time': start_times}, epochs=100, batch_size=32)