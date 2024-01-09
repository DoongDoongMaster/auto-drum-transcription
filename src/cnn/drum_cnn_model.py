import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

# from cnn.onset_detection import OnsetDetect
# import cnn.constant as constant

from onset_detection import OnsetDetect
import constant as constant

"""
-- 파라미터 & 값 지정
"""
# -- wav 길이가 다양하기 때문에, 길면 자르고 짧으면 padding 붙여서 일정하게 조절.
# -- 잘린 wav가 0.1sec~0.3sec이니까 20(0.2sec~0.3sec)으로 결정
max_pad_len = 20

# -- return 될 mfcc의 개수
n_mfcc_feature = 40

# -- model input shape
n_row = n_mfcc_feature
n_columns = max_pad_len
n_channels = 1

# -- 구분할 class 개수 8개
n_classes = 8

# -- epoch, learning_rate
training_epochs = 40
opt_learning_rate = 0.001

"""
-- 경로 지정
"""
# -- dataset path
input_file_path = "../../data"
# -- raw data path
root_path = input_file_path + "/raw_data"
# -- 저장될 onset detected drum data folder path
trim_path = input_file_path + "/trim_data"


onsetDetect = OnsetDetect(constant.SAMPLE_RATE, constant.ONSET_DURATION)

"""
-- librosa에서 추출한 audio의 feature 추출
"""


def extract_audio_feature(audio):
    mfccs = librosa.feature.mfcc(
        y=audio, sr=constant.SAMPLE_RATE, n_mfcc=n_mfcc_feature
    )
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    print(
        "-- length:",
        audio.shape[0] / float(constant.SAMPLE_RATE),
        "secs, ",
        "mfccs:",
        mfccs.shape,
    )
    return mfccs


"""
-- input data reshape
"""


def input_reshape(data):
    return tf.reshape(data, [-1, n_row, n_columns, n_channels])


"""
-- print
"""


def print_dataset_shape(x_train, y_train, x_val, y_val, x_test, y_test):
    print("x_train : ", x_train.shape)
    print("y_train : ", y_train.shape)
    print("x_val : ", x_val.shape)
    print("y_val : ", y_val.shape)
    print("x_test : ", x_test.shape)
    print("y_test : ", y_test.shape)


"""
-- 훈련(Train), 검증(Test) Dataset 생성
"""


def create_dataset(featuresdf):
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # -- split train, val, test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # -- print shape
    print_dataset_shape(x_train, y_train, x_val, y_val, x_test, y_test)

    # input shape 조정
    x_train = input_reshape(x_train)
    x_val = input_reshape(x_val)
    x_test = input_reshape(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


"""
-- trim 된 audio 가져오기
"""


def get_trimmed_audios(audio):
    # detect onsets
    onsets = onsetDetect.onset_detection(audio)
    # trimming audio
    trimmed_audios = onsetDetect.audio_trim_per_onset(audio, onsets)
    return trimmed_audios


"""
-- get train audio path
root_path
    └── dir/dir/...
        └── .m4a, .txt
"""
tmp_get_audio_path = []


def get_audio_path(root_path):
    datas = os.listdir(root_path)
    for d in datas:
        if d.endswith(".m4a") or d.endswith(".wav"):
            wav_path = os.path.join(root_path, d)
            tmp_get_audio_path.append(wav_path)
        elif d.endswith(".txt") == False:
            new_root_path = os.path.join(root_path, d)
            get_audio_path(new_root_path)
    result = tmp_get_audio_path
    return result


"""
-- class label mapping
"""


def get_class_label(idx, path):
    file_name = os.path.basename(path)
    if constant.PATTERN in path:  # -- pattern
        pattern_name = file_name[:2]  # -- P1
        class_label = constant.pattern2code[pattern_name][idx]
    elif constant.PER_DRUM in path:  # -- per drum
        drum_name = file_name[:2]  # -- CC
        class_label = constant.onehot_drum2code[drum_name]
    return class_label


"""
-- feature, label 갖고 오기
"""


def get_feature_label(audio_path_list):
    data_feature_label = []
    for path in audio_path_list:
        # -- librosa feature load
        audio, _ = librosa.load(path, sr=constant.SAMPLE_RATE, res_type="kaiser_fast")
        # -- trimmed audio
        trimmed_audios = get_trimmed_audios(audio)
        # -- trimmed feature
        for idx, taudio in enumerate(trimmed_audios):
            # -- mfcc
            trimmed_feature = extract_audio_feature(taudio)
            # -- class_label: 드럼 종류
            class_label = get_class_label(idx, path)
            data_feature_label.append([trimmed_feature, class_label])
    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(data_feature_label, columns=["feature", "class_label"])
    return featuresdf


"""
-- create model
"""


def create_model():
    model = keras.Sequential()

    model.add(
        layers.Conv2D(
            input_shape=(n_row, n_columns, n_channels),
            filters=16,
            kernel_size=(4, 4),
            activation="relu",
            padding="same",
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(
        layers.Conv2D(
            filters=16 * 2, kernel_size=(4, 4), activation="relu", padding="same"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(
        layers.Conv2D(
            filters=16 * 3, kernel_size=(4, 4), activation="relu", padding="same"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(units=n_classes, activation="sigmoid"))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=opt_learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


"""
-- evaluate model
"""


def evaluate_model(model, x_test, y_test):
    print("\n# Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=constant.batch_size)
    print("test loss:", results[0])
    print("test accuracy:", results[1])


def main():
    # -- train audio path 다 가져오기
    # -- : librosa.load를 위해선 path가 있어야 해서, path로 labeling 해야 해서
    audio_path_list = get_audio_path(root_path)
    print("--! get train audio path : ", len(audio_path_list), "개 !--")

    # -- feature, label
    featuresdf = get_feature_label(audio_path_list)
    print("--! get feature, label : ", featuresdf.shape, " !--")

    # -- create dataset
    x_train, x_val, x_test, y_train, y_val, y_test = create_dataset(featuresdf)

    # Create a new model instance
    model = create_model()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, mode="auto"
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=constant.batch_size,
        validation_data=(x_val, y_val),
        epochs=training_epochs,
        callbacks=[early_stopping],
    )

    stopped_epoch = early_stopping.stopped_epoch
    print("--! finish train : stopped_epoch >> ", stopped_epoch, " !--")
    evaluate_model(model, x_test, y_test)

    model.save(constant.checkpoint_path)
    print("--! save model !--")


if __name__ == "__main__":
    main()
