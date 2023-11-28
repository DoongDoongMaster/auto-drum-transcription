# -- onset 나누기
# -- file name : 본래wavfile이름_몇번째onset인지.wav
import os
from essentia.standard import MonoLoader, OnsetDetection, Windowing, FFT, CartesianToPolar, FrameGenerator, Onsets, AudioOnsetsMarker, StereoMuxer, AudioWriter
from tempfile import TemporaryDirectory
import scipy.io.wavfile
from essentia.standard import *
import scipy.io.wavfile

class OnsetDetect:
    def __init__(self, audio_sample_rate, onset_duration):
        self._audio_sample_rate = audio_sample_rate
        self._onset_duration = onset_duration
  
    # onset detection function
    def onset_detect(self, input_path):
        # Load audio file.
        audio = MonoLoader(filename=input_path)()

        # 1. Compute the onset detection function (ODF).

        # The OnsetDetection algorithm provides various ODFs.
        od_hfc = OnsetDetection(method='hfc')
        od_complex = OnsetDetection(method='complex')

        # We need the auxilary algorithms to compute magnitude and phase.
        w = Windowing(type='hann')
        fft = FFT() # Outputs a complex FFT vector.
        c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
            magnitude, phase = c2p(fft(w(frame)))
            pool.add('odf.hfc', od_hfc(magnitude, phase))
            pool.add('odf.complex', od_complex(magnitude, phase))

        # 2. Detect onset locations.
        onsets = Onsets()

        onsets_hfc = onsets(# This algorithm expects a matrix, not a vector.
                            essentia.array([pool['odf.hfc']]),
                            # You need to specify weights, but if we use only one ODF
                            # it doesn't actually matter which weight to give it
                            [1])

        onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])

        # onset 개수가 짝수인 것을 우선
        if len(onsets_hfc) % 2 == 0:
            return onsets_hfc
        elif len(onsets_complex) % 2 == 0:
            return onsets_complex
        elif len(onsets_hfc) > len(onsets_complex):
            return onsets_hfc
        else:
            return onsets_complex
        
    # trimmed audio
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

    # trimmed audio -> wav file write
    def write_trimmed_audio(self, root_path, name, trimmed_audios):
        start = 1
        for audio in trimmed_audios:
            # exist or not
            if not os.path.exists(root_path):
                # if the demo_folder directory is not present
                # then create it.
                os.makedirs(root_path)
            scipy.io.wavfile.write(f'{root_path}/{name}_{start:04}.wav', self._audio_sample_rate, audio)
            start += 1