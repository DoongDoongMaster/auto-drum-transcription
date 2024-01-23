import numpy as np
import pretty_midi
import xml.etree.ElementTree as ET

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


class OnsetDetect:
    """
    onset 구하는 클래스
    """

    @staticmethod
    def onset_detection(audio: np.ndarray) -> List[float]:
        """
        -- onset detection function. hfc method
        """
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

    @staticmethod
    def load_xml_data(file_path: str):
        """
        xml data 불러오기
        """
        try:
            tree = ET.parse(file_path)  # XML 파일을 파싱
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            print(f"XML 파일을 파싱하는 동안 오류가 발생했습니다: {e}")
            return None

    @staticmethod
    def get_onsets_from_xml(xml_path: str) -> List[float]:
        """
        -- XML file에서 onset 읽어오기
        """
        print("-- ! xml file location: ", xml_path)

        onset_sec_list = []
        xml_root = OnsetDetect.load_xml_data(xml_path)
        # transcription 엘리먼트의 정보 출력
        transcription_element = xml_root.find(".//transcription")
        events = transcription_element.findall("event")
        for event in events:
            onset_sec = event.find("onsetSec").text
            onset_sec_list.append(float(onset_sec))

        print("-- ! 파싱한 onsets: ", onset_sec_list)
        return onset_sec_list

    @staticmethod
    def get_onsets_from_txt(txt_path: str) -> List[float]:
        """
        -- TXT file에서 onset 읽어오기
        """
        print("-- ! txt file location: ", txt_path)

        onset_sec_list = []
        with open(txt_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                # 각 라인에서 onset 정보를 추출하여 리스트에 추가
                onset_sec = line.split()[0]
                onset_sec_list.append(float(onset_sec))

        print("-- ! 읽은 onsets: ", onset_sec_list)
        return onset_sec_list

    @staticmethod
    def get_onsets_from_mid(midi_path: str) -> List[float]:
        """
        -- MID file에서 onset 읽어오기
        """
        # MIDI 파일을 PrettyMIDI 객체로 로드합니다.
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # onset 정보를 저장할 리스트를 생성합니다.
        onset_times = []

        # 각 악기(track)에 대해 처리합니다.
        for instrument in midi_data.instruments:
            # 악기의 노트를 순회하며 onset을 찾습니다.
            for note in instrument.notes:
                onset_times.append(note.start)

        # onset_times를 정렬합니다.
        onset_times.sort()

        return onset_times
