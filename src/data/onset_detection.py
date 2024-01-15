import math

from typing import List

from essentia import Pool, array
from essentia.standard import (
    OnsetDetection,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
    Onsets,
)

"""
onset & 박자 구하는 클래스
"""


class OnsetDetect:
    def __init__(self, audio_sample_rate):
        self.audio_sample_rate = audio_sample_rate

    # onset detection function. hfc method
    def onset_detection(self, audio):
        # 1. Compute the onset detection function (ODF).
        od_hfc = OnsetDetection(method="hfc")

        # We need the auxilary algorithms to compute magnitude and phase.
        w = Windowing(type="hann")
        fft = FFT()  # Outputs a complex FFT vector.
        c2p = (
            CartesianToPolar()
        )  # Converts it into a pair of magnitude and phase vectors.

        # Compute both ODF frame by frame. Store results to a Pool.
        pool = Pool()
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
            magnitude, phase = c2p(fft(w(frame)))
            pool.add("odf.hfc", od_hfc(magnitude, phase))

        # 2. Detect onset locations.
        onsets = Onsets()
        onsets_hfc = onsets(  # This algorithm expects a matrix, not a vector.
            array([pool["odf.hfc"]]),
            # You need to specify weights, but if we use only one ODF
            # it doesn't actually matter which weight to give it
            [1],
        )

        print("-- ! onset ! --", onsets_hfc)
        return onsets_hfc

    """
    마디 단위별로 온셋 포인트 구하는 함수
    @param onset_full_audio: 전체 wav에 대한 onset time
    @param sec_of_bar: 한 마디가 몇 초인지
    @param bar_num: 마디 총 개수

    return 마디 단위별로의 온셋 time (마디의 처음 시작을 0, 끝을 1로 했을 때 기준으로)
    """

    def calculate_onset_per_bar(self, onset_full_audio, sec_of_bar, bar_num):
        onset_per_bar = [[] for i in range(bar_num)]  # 마디 개수만큼 row 초기화

        # 전체 wav에 대한 onset
        for onset in onset_full_audio:
            idx = math.floor(onset / sec_of_bar)  # 몇 번째 마디인지
            onset_point_in_bar = (
                onset - sec_of_bar * idx
            ) / sec_of_bar  # idx 마디에서 몇 박자 뒤에 등장하는지 (0 ~ 1)
            onset_per_bar[idx].append(onset_point_in_bar)

        return onset_per_bar

    """ 
    전체 wav와 bpm이 주어졌을 때, rhythm을 계산하는 함수
    @param audio_wav: wav array
    @param bpm: 분당 음표 개수 (4/4박자 기준)
    """

    def rhythm_detection(self, audio_wav, bpm, onset_full_audio):
        rhythm_per_bar = 4.0  # 한 마디에 4분음표가 몇 개 들어가는지
        rhythm_per_sec = bpm / 60.0  # 한 초당 몇 박 나오는지
        sec_of_bar = (1.0 / rhythm_per_sec) * rhythm_per_bar  # 한 마디가 몇 초인지
        bar_num = math.ceil(
            (len(audio_wav) / self.audio_sample_rate) / sec_of_bar
        )  # 총 마디 개수
        print(sec_of_bar, bar_num)

        bars = []  # 한 마디당 wav 정보
        step = (int)(sec_of_bar * self.audio_sample_rate)  # 한 마디에 들어가는 정보량
        for i in range(0, len(audio_wav), step):
            bars.append(audio_wav[i : i + step])

        onset_per_bar = self.calculate_onset_per_bar(
            onset_full_audio, sec_of_bar, bar_num
        )  # 한 마디당 rhythm 정보

        return onset_per_bar

    """ 
    전체 wav와 bpm이 주어졌을 때, 마디 별 음표의 박자를 계산하는 함수 
    @param audio_wav: wav array
    @param bpm: 분당 음표 개수 (4/4박자 기준)
    """

    def get_rhythm(
        self, audio_wav, bpm, onsets_arr: List[float] = None, is_our_train_data=False
    ):
        if is_our_train_data:  # 우리가 준비한 데이터셋이라면 앞에 공백 자르고 계산
            onset_full_audio = self.onset_detection(audio_wav)
            audio_wav = audio_wav[(int)(onset_full_audio[0] * self.audio_sample_rate) :]

        onset_full_audio = (
            onsets_arr if onsets_arr is not None else self.onset_detection(audio_wav)
        )  # 전체 wav에 대한 onset time

        if onset_full_audio is None:
            print("-- ! There is not exist onsets ! --")
            return

        return self.rhythm_detection(audio_wav, bpm, onset_full_audio)
