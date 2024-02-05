import librosa
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from typing import Dict, List
from essentia import Pool, array
from essentia.standard import (
    OnsetDetection,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
    Onsets,
)

from constant import SAMPLE_RATE


class OnsetDetect:
    """
    onset 구하는 클래스
    """

    @staticmethod
    def onset_detection(audio: np.ndarray) -> List[float]:
        """
        -- onset detection function. hfc method (using essentia library)
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
    def _get_filtering_onsets(
        onsets: List[float], start: float, end: float
    ) -> List[float]:
        """
        start ~ end 초 까지의 onsets 배열 구하는 함수
        """
        filter_onset = np.array(onsets)
        end = filter_onset[-1] + 1 if end == None else end
        filter_onset = filter_onset[(filter_onset >= start) & (filter_onset < end)]
        result = filter_onset - start  # start 빼서 0초부터 시작으로 맞추기

        np.set_printoptions(precision=2, suppress=True)
        print(f"-- ! {start} sec ~ {end} sec 파생한 onsets: ", result)
        return result

    @staticmethod
    def _load_xml_data(file_path: str):
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
    def get_onsets_from_xml(
        xml_path: str, start: float = 0, end: float = None
    ) -> List[float]:
        """
        -- XML file에서 onset 읽어오기
        """
        print("-- ! xml file location: ", xml_path)

        onset_sec_list = []
        xml_root = OnsetDetect._load_xml_data(xml_path)
        # transcription 엘리먼트의 정보 출력
        transcription_element = xml_root.find(".//transcription")
        events = transcription_element.findall("event")
        for event in events:
            onset_sec = event.find("onsetSec").text
            onset_sec_list.append(float(onset_sec))

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_sec_list, start, end)
        return onset_sec_list

    @staticmethod
    def get_onsets_instrument_from_xml(
        xml_path: str, start: float = 0, end: float = None, onset_dict={}
    ) -> Dict[str, List[float]]:
        """
        -- XML file에서 drum onsets 읽어오기

        * onset_dict: {'HH': [], 'ST': [], 'SD': [], 'KD': []}
        """
        print("-- ! xml file location: ", xml_path)

        xml_root = OnsetDetect._load_xml_data(xml_path)
        # transcription 엘리먼트의 정보 출력
        transcription_element = xml_root.find(".//transcription")
        events = transcription_element.findall("event")
        for event in events:
            onset_sec = float(event.find("onsetSec").text)
            drum_type = event.find("instrument").text

            if drum_type in onset_dict:
                onset_dict[drum_type].append(onset_sec)

        # Apply filtering
        for drum_type, onset_list in onset_dict.items():
            print("drum_type: ", drum_type, " ↴")
            onset_dict[drum_type] = OnsetDetect._get_filtering_onsets(
                onset_list, start, end
            )

        return onset_dict

    @staticmethod
    def _read_svl_file(file_path: str):
        """
        svl file 읽어오기
        """
        with open(file_path, "r", encoding="utf-8") as file:
            svl_content = file.read()
        return svl_content

    @staticmethod
    def get_onsets_from_svl(svl_path: str, start: float = 0, end: float = None):
        """
        -- svl file에서 onset 읽어오기
        """
        print("-- ! svl file location: ", svl_path)

        point_frames = []

        # Parse the XML content
        svl_content = OnsetDetect._read_svl_file(svl_path)
        root = ET.fromstring(svl_content)

        # Find all 'point' elements within 'dataset'
        dataset_element = root.find(".//dataset")
        if dataset_element is not None:
            point_elements = dataset_element.findall(".//point")

            # Extract frame attribute from each 'point' element
            for point_element in point_elements:
                frame_value = int(point_element.get("frame"))
                point_frames.append(frame_value)

        onset_sec_list = np.array(point_frames) / SAMPLE_RATE
        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_sec_list, start, end)
        return onset_sec_list

    @staticmethod
    def get_onsets_instrument_from_svl(
        svl_path: str, start: float = 0, end: float = None, onset_dict={}
    ) -> Dict[str, List[float]]:
        """
        -- svl file에서 onset 및 drum onsets 읽어오기

        * onset_dict: {'HH': [], 'ST': [], 'SD': [], 'KD': []}
        """

        # WaveDrum02_01#HH.svl -> HH 추출
        drum_type = svl_path[-6:-4]
        if drum_type == "KD":
            drum_type = "KK"

        if drum_type in onset_dict:
            print("drum_type: ", drum_type, " ↴")
            onset_dict[drum_type] = OnsetDetect.get_onsets_from_svl(
                svl_path, start, end
            )

        return onset_dict

    @staticmethod
    def get_onsets_from_txt(
        txt_path: str, start: float = 0, end: float = None
    ) -> List[float]:
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

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_sec_list, start, end)
        return onset_sec_list

    @staticmethod
    def get_onsets_from_mid(
        midi_path: str, start: float = 0, end: float = None
    ) -> List[float]:
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

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list

    @staticmethod
    def get_peak_using_librosa(
        audio: np.ndarray,
        hop_length: int,
        bpm: int = 100,
        start: float = 0,
        end: float = None,
    ) -> List[float]:
        """
        -- librosa 라이브러리 사용해서 peak 구하기
        """
        rhythm_per_bar = 4.0  # 한 마디에 4분음표가 몇 개 들어가는지
        sec_per_note = 60.0 / float(bpm)  # 4분음표 하나가 몇 초
        wait = (sec_per_note * rhythm_per_bar) / float(
            rhythm_per_bar * 8
        )  # 음표 간격 최소 단위 (32bit까지 나오는 기준)

        onset_env = librosa.onset.onset_strength(
            y=audio, sr=SAMPLE_RATE, hop_length=hop_length, aggregate=np.median
        )
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=5, wait=wait
        )
        onset_times = librosa.frames_to_time(
            peaks, sr=SAMPLE_RATE, hop_length=hop_length
        )

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list

    @staticmethod
    def _show_onset_plot(onset_env, onset_frames):
        plt.plot(onset_env, alpha=0.8)
        plt.plot(onset_frames, onset_env[onset_frames], "x")
        plt.legend(frameon=True, framealpha=0.8)
        plt.show()

    @staticmethod
    def get_onsets_using_librosa(
        audio: np.ndarray,
        hop_length: int,
        start: float = 0,
        end: float = None,
    ) -> List[float]:
        """
        -- librosa 라이브러리 사용해서 onset 구하기
        """
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=SAMPLE_RATE, hop_length=hop_length, lag=1
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=SAMPLE_RATE, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=SAMPLE_RATE, hop_length=hop_length
        )

        OnsetDetect._show_onset_plot(onset_env, onset_frames)

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list
