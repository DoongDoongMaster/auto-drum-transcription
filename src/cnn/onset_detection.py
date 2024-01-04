# -- onset 나누기
# -- file name : 본래wavfile이름_몇번째onset인지.wav
import os
import numpy as np
from essentia.standard import OnsetDetection, Windowing, FFT, CartesianToPolar, FrameGenerator, Onsets
import scipy.io.wavfile
from essentia.standard import *
import scipy.io.wavfile
import math

class OnsetDetect:
    def __init__(self, audio_sample_rate, onset_duration):
        self._audio_sample_rate = audio_sample_rate
        self._onset_duration = onset_duration
  
    # onset detection function
    def onset_detection(self, audio):
        # 1. Compute the onset detection function (ODF).

        # The OnsetDetection algorithm provides various ODFs.
        od_hfc = OnsetDetection(method='hfc')

        # We need the auxilary algorithms to compute magnitude and phase.
        w = Windowing(type='hann')
        fft = FFT() # Outputs a complex FFT vector.
        c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
            magnitude, phase = c2p(fft(w(frame)))
            pool.add('odf.hfc', od_hfc(magnitude, phase))

        # 2. Detect onset locations.
        onsets = Onsets()

        onsets_hfc = onsets(# This algorithm expects a matrix, not a vector.
                            essentia.array([pool['odf.hfc']]),
                            # You need to specify weights, but if we use only one ODF
                            # it doesn't actually matter which weight to give it
                            [1])

        return onsets_hfc
        
    # trimmed audio per onset -> list
    def audio_trim_per_onset(self, audio, onsets):
        sr = self._audio_sample_rate
        duration = self._onset_duration

        trimmed_audios = []
        for i in range(0, len(onsets)):
            start = (int)((onsets[i]) * sr)
            if i + 1 < len(onsets):
                end_by_onset = (int)(onsets[i + 1] * sr)
            end_by_duration = (int)((onsets[i] + duration) * sr)

            if i + 1 < len(onsets):
                end = min(end_by_onset, end_by_duration)
            else:
                end = end_by_duration

            trimmed = audio[start:end]
            trimmed_audios.append(trimmed)

        return trimmed_audios
    
    # trimmed audio from first onset to last audio (sequece data)
    def audio_trim_first_onset(self, audio, first_onset):
        sr = self._audio_sample_rate
        start = (int)(first_onset * sr)
        trimmed = audio[start:]
        return trimmed
    
    # trimmed audio -> wav file write
    def write_trimmed_audio_one(self, root_path, name, trimmed_audio):
        # exist or not
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        scipy.io.wavfile.write(f'{root_path}/{name}.wav', self._audio_sample_rate, trimmed_audio)

    # trimmed audio -> wav file write list
    def write_trimmed_audio(self, root_path, name, trimmed_audios):
        start = 1
        for audio in trimmed_audios:
            self.write_trimmed_audio_one(root_path, f'{name}_{start:04}', audio)
            start += 1
    
    """
    마디 단위별로 온셋 포인트 구하는 함수
    @param onset_full_audio: 전체 wav에 대한 onset time
    @param sec_of_bar: 한 마디가 몇 초인지
    @param bar_num: 마디 총 개수

    return 마디 단위별로의 온셋 time (마디의 처음 시작을 0, 끝을 1로 했을 때 기준으로)
    """
    def calculate_onset_per_bar(self, onset_full_audio, sec_of_bar, bar_num):
        onset_per_bar = [[] for i in range(bar_num)] # 마디 개수만큼 row 초기화

        # 전체 wav에 대한 onset 
        for onset in onset_full_audio:
            idx = math.floor(onset / sec_of_bar) # 몇 번째 마디인지
            onset_point_in_bar = (onset - sec_of_bar * idx) / sec_of_bar # idx 마디에서 몇 박자 뒤에 등장하는지 (0 ~ 1)
            onset_per_bar[idx].append(onset_point_in_bar)
    
        return onset_per_bar
    
    """ 
    전체 wav와 bpm이 주어졌을 때, rhythm을 계산하는 함수
    @param audio_wav: wav array
    @param bpm: 분당 음표 개수 (4/4박자 기준)
    """
    def rhythm_detection(self, audio_wav, bpm):
        rhythm_per_bar = 4.0 # 한 마디에 4분음표가 몇 개 들어가는지
        rhythm_per_sec = bpm / 60.0 # 한 초당 몇 박 나오는지
        sec_of_bar = (1.0 / rhythm_per_sec) * rhythm_per_bar # 한 마디가 몇 초인지
        bar_num = math.ceil((len(audio_wav) / self._audio_sample_rate) / sec_of_bar) # 총 마디 개수
        print(sec_of_bar, bar_num)

        bars = [] # 한 마디당 wav 정보
        step = (int)(sec_of_bar * self._audio_sample_rate) # 한 마디에 들어가는 정보량
        for i in range(0, len(audio_wav), step):
            bars.append(audio_wav[i:i+step])

        onset_full_audio = self.onset_detection(audio_wav) # 전체 wav에 대한 onset time
        onset_per_bar = self.calculate_onset_per_bar(onset_full_audio, sec_of_bar, bar_num) # 한 마디당 rhythm 정보
        
        return onset_per_bar
    
    def get_rhythm(self, audio_wav, bpm, is_our_train_data=False):
        if is_our_train_data:
            onset_full_audio = self.onset_detection(audio_wav)
            audio_wav = audio_wav[(int)(onset_full_audio[0] * self._audio_sample_rate):]
        
        return self.rhythm_detection(audio_wav, bpm)
    
    def manage_delay_time(self, audio, delay_micro_sec):
        start_point = delay_micro_sec * (self._audio_sample_rate / 1000000) 
        if delay_micro_sec < 0:
            start_point = start_point * (-1)
            start_empty_list = np.array([0.0] * (int)(start_point), dtype=np.float32)
            result = np.concatenate((start_empty_list, audio), axis=0)
        else:
            result = audio[(int)(start_point):]
        return result