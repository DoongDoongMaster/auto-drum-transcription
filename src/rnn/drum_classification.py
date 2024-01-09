import os
import sys

import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import constant 
import drum_rnn_model

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cnn.onset_detection as onset_detection


onsetDetect = onset_detection.OnsetDetect(constant.SAMPLE_RATE, constant.ONSET_DURATION)
predict_model = tf.keras.models.load_model(constant.checkpoint_path)


def get_predict_result(audio, bpm):
    # translate STFT
    stft = librosa.stft(y=audio, n_fft=constant.N_FFT, hop_length=constant.HOP_LENGTH, win_length=constant.WIN_LENGTH, window='hann')
    stft = np.abs(stft, dtype=np.float64)
    if stft.shape[1] < constant.TIME_WIDTH:
      stft_new = np.pad(stft, pad_width=((0,0), (0, constant.TIME_WIDTH - stft.shape[1])), mode='constant')
    else:
      stft_new = stft[:, :constant.TIME_WIDTH]
    stft_new = np.transpose(stft_new)

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

    x_predict = stft_new.reshape((1, stft_new.shape[0], stft_new.shape[1]))
    prediction = predict_model.predict(x_predict)
    predicted = prediction[0]

    plt.plot(predicted)
    plt.title('Model prediction result')
    plt.show()
    
    return predicted


def get_drum_data(wav_path, bpm, delay):
    audio, _ = librosa.load(wav_path, sr=constant.SAMPLE_RATE)
    new_audio = onsetDetect.manage_delay_time(audio, delay)

    # -- instrument & rhythm
    result = get_predict_result(new_audio, bpm)
    
    return result


def main():
    wav_path='../../data/sequence_data/pattern/P1/08/P1_08_0001.wav'
    bpm = 100
    delay = 0
    result = get_drum_data(wav_path, bpm, delay)
    print(result)


if __name__ == "__main__":
    main()
