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

from onset_detection import OnsetDetect
import constant

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
input_file_path = '../../data'
# -- raw data path
root_path = input_file_path + '/raw_data'
# -- 저장될 onset detected drum data folder path
trim_path = input_file_path + '/trim_data'


"""
-- feature 추출
"""
def extract_feature(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=constant.SAMPLE_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc_feature)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        print('file name :', file_name, ', length:', audio.shape[0]/float(sample_rate), 'secs, ', 'mfccs:', mfccs.shape)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None
    return mfccs

"""
-- onset Data 가져와서 feature 추출
"""
data_feature_label = []
def extract_trim_feature(root_path):
    datas = os.listdir(root_path)

    for d in datas:
        path = os.path.join(root_path, d)
        if d.endswith('.wav'):
            wav = path
            # -- feature: mfcc
            feature = extract_feature(wav)
            # -- class_label: 드럼 종류
            file_name = d[:-4]
            if constant.PATTERN in path: # -- pattern
                pattern_name = file_name[:2] # -- P1
                drum_name = file_name[-4:] # -- 0001
                class_label = constant.pattern2code[pattern_name][drum_name]
            elif constant.PER_DRUM in path: # -- per drum
                class_label = constant.onehot_drum2code[file_name[:2]] # -- CC

            data_feature_label.append([feature, class_label])

        elif os.path.isdir(path):
            new_root_path = os.path.join(root_path, d)
            extract_trim_feature(new_root_path)

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(data_feature_label, columns=['feature','class_label'])
    return featuresdf

def input_reshape(data):
    return tf.reshape(data, [-1, n_row, n_columns, n_channels])

"""
-- 훈련(Train), 검증(Test) Dataset 생성
"""
def create_dataset(featuresdf):
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    yy= y

    # -- train, test 분류
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)

    print("one-hot-encoding 전 : ", y[:5])
    print("one-hot-encoding 후 : ", yy[:5])

    print("x_train : ", x_train.shape)
    print("y_train : ", y_train.shape)
    print("x_train : ", x_val.shape)
    print("y_train : ", y_val.shape)
    print("x_test : ", x_test.shape)
    print("y_test : ", y_test.shape)

    # input shape 조정
    x_train = input_reshape(x_train)
    x_val = input_reshape(x_val)
    x_test = input_reshape(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

def create_model():
    model = keras.Sequential()

    model.add(layers.Conv2D(input_shape=(n_row, n_columns, n_channels), filters=16, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=16*2, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=16*3, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(units=n_classes, activation='sigmoid'))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=opt_learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def create_trim_data(root_path, trim_path):
  onsetDetect = OnsetDetect(constant.SAMPLE_RATE, constant.ONSET_DURATION)
  datas = os.listdir(root_path)

  for d in datas:
    if d.endswith('.m4a') or d.endswith('.wav'):
        wav = os.path.join(root_path, d)

        # detect onsets
        onsets = onsetDetect.onset_detect(wav)
        # trimming audio
        audio, sr = librosa.load(wav, sr=constant.SAMPLE_RATE)
        trimmed_audios = onsetDetect.audio_trim_per_onset(audio, onsets)
        # new_file write
        name = d[:-4]
        onsetDetect.write_trimmed_audio(trim_path, name, trimmed_audios)

    elif d.endswith('.txt') == False:
        new_root_path = os.path.join(root_path, d)
        new_trim_path = os.path.join(trim_path, d)
        create_trim_data(new_root_path, new_trim_path)

def evaluate_model(model, x_test, y_test):
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=constant.batch_size)
    print('test loss:', results[0])
    print('test accuracy:', results[1])

def main():
    # create_trim_data(root_path, trim_path)

    featuresdf = extract_trim_feature(trim_path)

    x_train, x_val, x_test, y_train, y_val, y_test = create_dataset(featuresdf)

    # ModelCheckpoint callback to save weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=constant.checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Create a new model instance
    model = create_model()
    # Save the weights using the `checkpoint_path` format
    model.save_weights(constant.checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode = 'auto')
    history = model.fit(x_train, y_train, batch_size=constant.batch_size, validation_data = (x_val, y_val), epochs=training_epochs, callbacks=[early_stopping, checkpoint_callback])
    stopped_epoch = early_stopping.stopped_epoch
    print("stopped_epoch >> ", stopped_epoch)

    evaluate_model(model, x_test, y_test)
    model.save(constant.save_model_path)

if __name__ == "__main__":
    main()