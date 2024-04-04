import os
import librosa
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from typing import List, Dict
from datetime import datetime
from essentia import Pool, array
from essentia.standard import (
    OnsetDetection,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
    Onsets,
)

from constant import (
    DDM_OWN,
    DRUM_KIT,
    DRUM_MAP,
    IDMT,
    SAMPLE_RATE,
    IMAGE_PATH,
    DRUM2CODE,
    DRUM_TYPES,
)


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
        od_complex = OnsetDetection(method="complex")

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
            pool.add("odf.complex", od_complex(magnitude, phase))

        # 2. Detect onset locations.
        onsets = Onsets()
        onsets_hfc = onsets(  # This algorithm expects a matrix, not a vector.
            array([pool["odf.hfc"]]),
            # You need to specify weights, but if we use only one ODF
            # it doesn't actually matter which weight to give it
            [1],
        )
        onsets_complex = onsets(array([pool["odf.complex"]]), [1])

        n_frames = len(pool["odf.complex"])
        frames_position_samples = np.array(range(n_frames)) * 512

        fig, ((ax1, ax3)) = plt.subplots(
            2, 1, sharex=True, sharey=False, figsize=(15, 16)
        )

        ax1.set_title("complex ODF")
        ax1.plot(frames_position_samples, pool["odf.complex"], color="magenta")

        ax3.set_title("Audio waveform and the estimated onset positions (complex ODF)")
        ax3.plot(audio)
        for onset in onsets_complex:
            ax3.axvline(x=onset * SAMPLE_RATE, color="magenta")

        # plt.show()

        # print("-- ! onset ! --", onsets_complex)
        return onsets_complex

    @staticmethod
    def _get_filtering_onsets(
        onsets: List[float], start: float, end: float
    ) -> List[float]:
        """
        start ~ end 초 까지의 onsets 배열 구하는 함수
        """
        filter_onset = np.array(onsets)
        if filter_onset.size == 0:
            return filter_onset

        end = filter_onset[-1] + 1 if end == None else end
        filter_onset = filter_onset[(filter_onset >= start) & (filter_onset < end)]

        result = filter_onset - start  # start 빼서 0초부터 시작으로 맞추기

        np.set_printoptions(precision=3, suppress=True)
        print(f"-- ! {start} sec ~ {end} sec 파생한 onsets: ", result)
        return result

    @staticmethod
    def _get_filtering_onsets_instrument(
        onsets: dict[str, List[float]], start: float, end: float
    ) -> dict[str, List[float]]:
        """
        start ~ end 초 까지의 instrument별 onsets 배열 구하는 함수
        """
        result = {drum: [] for drum in onsets}
        for drum, onsets_arr in onsets.items():
            print(f"-- ! {drum} 악기의 onset ! --")
            result[drum] = OnsetDetect._get_filtering_onsets(onsets_arr, start, end)
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

        * onset_dict: {'CC':[], 'OH':[], 'CH':[], 'TT':[], 'SD':[], 'HH':[]}
        """
        print("-- ! xml file location: ", xml_path)

        xml_root = OnsetDetect._load_xml_data(xml_path)
        # transcription 엘리먼트의 정보 출력
        transcription_element = xml_root.find(".//transcription")
        events = transcription_element.findall("event")
        for event in events:
            onset_sec = float(event.find("onsetSec").text)
            drum_type = event.find("instrument").text
            drum_type = DRUM_MAP[drum_type]
            if drum_type in onset_dict:
                onset_dict[drum_type].append(onset_sec)

        onset_dict = OnsetDetect._get_filtering_onsets_instrument(
            onset_dict, start, end
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

        * onset_dict: {'CC':[], 'OH':[], 'CH':[], 'TT':[], 'SD':[], 'HH':[]}
        """

        # WaveDrum02_01#HH.svl -> HH 추출
        drum_type = svl_path[-6:-4]
        drum_type = DRUM_MAP[drum_type]

        if drum_type in onset_dict:
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
    def get_onsets_instrument_from_txt(
        txt_path: str, start: float = 0, end: float = None, onset_dict={}
    ) -> List[float]:
        """
        -- TXT file에서 onset 읽어오기

        * onset_dict: {'CC':[], 'OH':[], 'CH':[], 'TT':[], 'SD':[], 'HH':[]}
        """
        print("-- ! txt file location: ", txt_path)

        # Open the text file
        with open(txt_path, "r") as file:
            lines = file.readlines()

        # -- ex) 2.674 mt => 2.674 / mt->ST
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                onset_sec = float(parts[0])
                drum_type = parts[1]
                drum_type = DRUM_MAP[drum_type]

                if drum_type in onset_dict:
                    onset_dict[drum_type].append(onset_sec)

        onset_dict = OnsetDetect._get_filtering_onsets_instrument(
            onset_dict, start, end
        )
        return onset_dict

    @staticmethod
    def get_onsets_instrument_all_from_txt(txt_path: str) -> List[dict]:
        """
        -- txt file에서 {”onset”: onset, “drum”: 악기} 를 원소로 가진 List 형태로 반환하는 함수
        """
        # Open the text file
        with open(txt_path, "r") as file:
            lines = file.readlines()

        # -- ex) 2.674 mt => 2.674 / mt->ST
        result_onsets = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                onset_sec = float(parts[0])
                drum_type = parts[1]

                data = {"onset": onset_sec, "drum": -1}
                if drum_type in DRUM_MAP:
                    data["drum"] = DRUM2CODE[DRUM_MAP[drum_type]]
                result_onsets.append(data)

        return result_onsets

    @staticmethod
    def _get_drum_track_from_mid(midi_data):
        # Find the drum track
        drum_track = next(
            (instrument for instrument in midi_data.instruments if instrument.is_drum),
            None,
        )

        if drum_track is None:
            print("No drum track found in the MIDI file.")
            return None

        return drum_track

    @staticmethod
    def get_onsets_from_mid(
        midi_path: str, start: float = 0, end: float = None
    ) -> List[float]:
        """
        -- MID file에서 onset 읽어오기
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        drum_track = OnsetDetect._get_drum_track_from_mid(midi_data)
        onset_times = [note.start for note in drum_track.notes]
        onset_times.sort()
        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list

    @staticmethod
    def get_onsets_instrument_from_mid(
        midi_path: str, start: float = 0, end: float = None, onset_dict={}
    ) -> dict[str, List[float]]:
        """
        -- midi file에서 드럼 악기별로 onset을 구하는 함수

        onset_dict : {'CC':[], 'OH':[], 'CH':[], 'TT':[], 'SD':[], 'HH':[]}
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        drum_track = OnsetDetect._get_drum_track_from_mid(midi_data)

        # Dictionary to store onsets for each selected drum instrument
        drum_onsets = onset_dict

        # 악기의 노트를 순회하며 onset을 찾음
        for note in drum_track.notes:
            if note.pitch in DRUM_MAP:
                drum_onsets[DRUM_MAP[note.pitch]].append(note.start)

        drum_onsets = OnsetDetect._get_filtering_onsets_instrument(
            drum_onsets, start, end
        )
        return drum_onsets

    @staticmethod
    def get_onsets_instrument_all_from_mid(midi_path: str) -> List[dict]:
        """
        -- midi file에서 {”onset”: onset, “drum”: 악기} 를 원소로 가진 List 형태로 반환하는 함수
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        drum_track = OnsetDetect._get_drum_track_from_mid(midi_data)

        result_onsets = []
        # 악기의 노트를 순회하며 onset을 찾음
        for note in drum_track.notes:
            data = {"onset": note.start, "drum": -1}
            if note.pitch in DRUM_MAP:
                data["drum"] = DRUM2CODE[DRUM_MAP[note.pitch]]
            result_onsets.append(data)

        return result_onsets

    @staticmethod
    def get_onsets_instrument_from_wav(
        audio: np.ndarray,
        wav_path: str,
        start: float = 0,
        end: float = None,
        onset_dict: dict[str, List[float]] = {},
    ):
        """
        -- wav file에서 악기별로 onset을 가져오는 함수 (drum_kit, ddm data에서 사용)
        drum_kit : 실제로 둥 한 번 치는 데이터인데, librosa에서 onset이 여러 개 추출돼서 onset[0]만 데이터로 사용
        ddm : librosa에서 추출된 그대로 데이터로 사용
        """
        onsets = OnsetDetect.onset_detection(audio)

        drum_type = ""
        is_drum_kit = False
        is_ddm = False
        is_ddm_cc_08 = False  # 08박자면 처음 한 번만 치므로 예외처리
        if DDM_OWN in wav_path:  # ddm own
            is_ddm = True
            drum_type = wav_path.split("/")[-3]
            rhythm = wav_path.split("/")[-2]
            if drum_type == "CC" and rhythm == "08":
                is_ddm_cc_08 = True
        elif DRUM_KIT in wav_path:  # drum kit
            is_drum_kit = True
            drum_type = wav_path.split("/")[-2]
        elif IDMT in wav_path:  # idmt
            drum_type = wav_path.split("#")[1]  # 'KD', 'SD', 'HH'

        if drum_type in DRUM_MAP:
            if is_drum_kit:
                onsets = [0.01]
            elif is_ddm_cc_08:
                onsets = [onsets[(onsets >= 2.0) & (onsets < 3.0)][0]]
            elif is_ddm:
                onsets = onsets[(onsets >= 2.0) & (onsets < 5.1)]
            else:
                onsets = onsets
            onset_dict[DRUM_MAP[drum_type]] = onsets

        onset_dict = OnsetDetect._get_filtering_onsets_instrument(
            onset_dict, start, end
        )
        return onset_dict

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
            onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=1,
            wait=wait,
        )
        onset_times = librosa.frames_to_time(
            peaks, sr=SAMPLE_RATE, hop_length=hop_length
        )

        # OnsetDetect._show_onset_plot(onset_env, peaks)

        onset_times = onset_times - 0.01

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list

    @staticmethod
    def _show_onset_plot(onset_env, onset_frames):
        plt.plot(onset_env, alpha=0.8)
        plt.plot(onset_frames, onset_env[onset_frames], "x")
        plt.legend(frameon=True, framealpha=0.8)
        os.makedirs(IMAGE_PATH, exist_ok=True)  # 이미지 폴더 생성
        date_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # 현재 날짜와 시간 가져오기
        plt.savefig(f"{IMAGE_PATH}/label-{date_time}.png")
        plt.show()

    @staticmethod
    def get_onsets_using_librosa(
        audio: np.ndarray,
        start: float = 0,
        end: float = None,
        hop_length: int = 441,
    ) -> List[float]:
        """
        -- librosa 라이브러리 사용해서 onset 구하기
        """
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=SAMPLE_RATE,
            hop_length=hop_length,
            lag=1,
            aggregate=np.median,
            n_fft=2048,
            fmax=8000,
            n_mels=256,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=SAMPLE_RATE, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=SAMPLE_RATE, hop_length=hop_length
        )
        onset_times = onset_times - 0.01

        # OnsetDetect._show_onset_plot(onset_env, onset_frames)

        onset_sec_list = OnsetDetect._get_filtering_onsets(onset_times, start, end)
        return onset_sec_list
