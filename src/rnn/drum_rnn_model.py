import os
import sys
import librosa

import random
import numpy as np
import tensorflow as tf
from essentia.standard import MonoLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from cnn.onset_detection import OnsetDetect
import constant

"""
-- 파라미터 & 값 지정
"""
# -- epoch, learning_rate
training_epochs = 10
opt_learning_rate = 0.01

"""
-- 경로 지정
"""
# -- dataset path
data_file_path = '../../data'
root_path = data_file_path + "/raw_data/"
trim_path = data_file_path + "/sequence_data/"
data_root_path = f"{data_file_path}/sequence_data/per_drum/{constant.DRUM}_{constant.code2drum[constant.DRUM]}"

onsetDetect = OnsetDetect(constant.SAMPLE_RATE, constant.ONSET_DURATION)

# load raw data & trimming sequence data & save
def load_and_write_audio(root_path, trim_path):
    datas = os.listdir(root_path)

    for d in datas:
        if d.endswith('.m4a') or d.endswith('.wav'):
            wav_path = os.path.join(root_path, d)

            # detect onsets
            audio = MonoLoader(filename=wav_path)()
            onsets = onsetDetect.onset_detection(audio)

            # trimming audio
            trimmed_audios = onsetDetect.audio_trim_first_onset(audio, onsets[0])
            # print(trimmed_audios)

            # new_file write
            name = d[:-4]
            onsetDetect.write_trimmed_audio_one(trim_path, name, trimmed_audios)
        elif os.path.isdir(os.path.join(root_path, d)): # folder인 경우
            new_root_path = os.path.join(root_path, d)
            new_trim_path = os.path.join(trim_path, d)
            load_and_write_audio(new_root_path, new_trim_path)

# load sequence data per drum
def load_sequence_data_per_drum(data_root_path):
    beat_folders = os.listdir(data_root_path)

    X_data_full = []
    for beat in beat_folders:
        X_data_full += [os.path.join(data_root_path, beat, file) for file in os.listdir(os.path.join(data_root_path, beat))]

    random.shuffle(X_data_full)
    return X_data_full

# load sequence data all
def load_sequence_data_all(root_path):
    if os.path.isfile(root_path):
        if root_path.endswith('wav') or root_path.endswith('m4a'):
            return [root_path]
        else:
            return []  

    folders = os.listdir(root_path)
    audio_paths = []

    for d in folders:
        new_root_path = os.path.join(root_path, d)
        audio_paths += load_sequence_data_all(new_root_path)
        
    return audio_paths

# data labeling per drum
def data_labeling_per_drum(datas):
    X_data = []
    Y_data = []

    # data load
    for path in datas:
        # load audio wav array
        audio = MonoLoader(filename=path)()

        # translate STFT
        stft = librosa.stft(y=audio, n_fft=constant.N_FFT, hop_length=constant.HOP_LENGTH, win_length=constant.WIN_LENGTH, window='hann')
        stft = np.abs(stft, dtype=np.float64)
        if stft.shape[1] < constant.TIME_WIDTH:
            stft_new = np.pad(stft, pad_width=((0,0), (0, constant.TIME_WIDTH - stft.shape[1])), mode='constant')
        else:
            stft_new = stft[:, :constant.TIME_WIDTH]
        stft_new = np.transpose(stft_new)
        X_data.append(stft_new)

        print("--! data labeling !--", stft_new)

        # onset detection & labeling
        # onset position : 1
        # onset position with ONSET_DURATION : 0.5
        # extra : 0
        onsets_arr = onsetDetect.onset_detection(audio)
        labels = [0.0] * constant.TIME_WIDTH

        for onset in onsets_arr:
            soft_start_position = max((onset - constant.ONSET_DURATION), 0) * constant.SAMPLE_RATE / constant.HOP_LENGTH
            onset_position = onset * constant.SAMPLE_RATE / constant.HOP_LENGTH
            soft_end_position = (onset + constant.ONSET_DURATION) * constant.SAMPLE_RATE / constant.HOP_LENGTH
            for i in range((int)(soft_start_position), (int)(soft_end_position)):
                if labels[i] == 1:
                    continue
                labels[i] = 0.5
            labels[(int)(onset_position)] = 1

        Y_data.append(labels)

    return np.array(X_data), np.array(Y_data)

# data labeling all drum
def data_labeling_all(datas):
    X_data = []
    Y_data = []

    # data load
    for path in datas:
        # load audio wav array
        audio = MonoLoader(filename=path)()
        file_name = path[-14:]
        file_name = file_name[:-4]

        # translate STFT
        stft = librosa.stft(y=audio, n_fft=constant.N_FFT, hop_length=constant.HOP_LENGTH, win_length=constant.WIN_LENGTH, window='hann')
        stft = np.abs(stft, dtype=np.float64)
        if stft.shape[1] < constant.TIME_WIDTH:
            stft_new = np.pad(stft, pad_width=((0,0), (0, constant.TIME_WIDTH - stft.shape[1])), mode='constant')
        else:
            stft_new = stft[:, :constant.TIME_WIDTH]
        stft_new = np.transpose(stft_new)
        X_data.append(stft_new)

        print("--! data labeling !--", stft_new)

        # onset detection & labeling
        # onset position : 1
        # onset position with ONSET_DURATION : 0.5
        # extra : 0
        onsets_arr = onsetDetect.onset_detection(audio)
        labels = [[0.0] * len(constant.code2drum) for _ in range(constant.TIME_WIDTH)]
        pattern_idx = 0
        for onset in onsets_arr:
            soft_start_position = max((onset - constant.ONSET_DURATION), 0) * constant.SAMPLE_RATE / constant.HOP_LENGTH
            onset_position = onset * constant.SAMPLE_RATE / constant.HOP_LENGTH
            soft_end_position = (onset + constant.ONSET_DURATION) * constant.SAMPLE_RATE / constant.HOP_LENGTH
            if any(drum in file_name for idx, drum in constant.code2drum.items()): # per drum
                one_hot_label = constant.onehot_drum2code[file_name[:2]]
            else: # pattern
                pattern_name = file_name[:2] # -- P1
                one_hot_label = constant.pattern2code[pattern_name][pattern_idx]
                pattern_idx += 1
            for i in range((int)(soft_start_position), (int)(soft_end_position)):
                if (np.array(labels[i]) == np.array(one_hot_label)).all():
                    continue
                labels[i] = np.array(one_hot_label) / 2
            labels[(int)(onset_position)] = one_hot_label

        Y_data.append(labels)

    return np.array(X_data), np.array(Y_data)

# split data set (train & test)
def split_data_set(X_data, Y_data):
    # split train & test dataset
    test_cnt = (int)(X_data.shape[0] * 0.2)

    X_train, Y_train = X_data[:-test_cnt], Y_data[:-test_cnt]
    X_test, Y_test = X_data[-test_cnt:], Y_data[-test_cnt:]

    return X_train, Y_train, X_test, Y_test


def create_model_all():
    # build the model
    model = tf.keras.Sequential()

    # bidirection-RNN layer
    unit_number = 50
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(unit_number, return_sequences=True, input_shape=(constant.TIME_WIDTH, constant.N_FFT // 2 + 1), activation='relu')))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(unit_number, return_sequences=True, activation='relu')))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(unit_number, return_sequences=True, activation='relu')))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # dense layer
    model.add(tf.keras.layers.Dense(constant.TIME_WIDTH * len(constant.code2drum), activation='sigmoid'))

    model.build((None, constant.TIME_WIDTH, constant.N_FFT // 2 + 1))
    model.summary()

    # compile the model
    opt = tf.keras.optimizers.SGD(learning_rate=opt_learning_rate, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_model_one():
    # build the model
    model = tf.keras.Sequential()

    # bidirection-RNN layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=(constant.TIME_WIDTH, constant.N_FFT // 2 + 1), activation='relu')))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50, return_sequences=True, activation='relu')))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50, return_sequences=True, activation='relu')))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # dense layer
    model.add(tf.keras.layers.Dense(constant.TIME_WIDTH, activation='sigmoid'))

    model.build((None, constant.TIME_WIDTH, constant.N_FFT // 2 + 1))
    model.summary()

    # compile the model
    opt = tf.keras.optimizers.SGD(learning_rate=opt_learning_rate, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def evaluate_model(model, x_test, y_test):
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=constant.batch_size)
    print('test loss:', results[0])
    print('test accuracy:', results[1])


def main():
    # data sequence로 잘라서 새로 저장
    load_and_write_audio(root_path, trim_path)

    # 데이터 라벨링 (한 악기 씩 | 여러 악기 한 번에)
    X_data, Y_data = [], []
    if constant.MODE == 0:
        X_data_full = load_sequence_data_per_drum(data_root_path)
        X_data, Y_data = data_labeling_per_drum(X_data_full)
    else:
        X_data_full = load_sequence_data_all(trim_path)
        X_data, Y_data = data_labeling_all(X_data_full[:200])

    # 데이터셋 분리
    X_train, Y_train, X_test, Y_test = split_data_set(X_data, Y_data)

    # Create a new model instance
    model = None
    if constant.MODE == 0:
        model = create_model_one()
    else:
        model = create_model_all()

    model.fit(X_train, Y_train, batch_size=constant.batch_size, epochs=training_epochs)
    print("--! finish train !-- ")
    
    evaluate_model(model, X_test, Y_test)

    model.save(constant.checkpoint_path)
    print("--! save model !--")


if __name__ == "__main__":
    main()