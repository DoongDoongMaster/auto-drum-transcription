"""
-- image path (그래프와 같은 이미지 저장)
"""

IMAGE_PATH = "../images"


"""
-- data path
"""
ROOT_PATH = "../data"
RAW_PATH = "raw"
NEW_PATH = "new"
PROCESSED_FEATURE = "processed-feature"


"""
-- data origin (데이터 출처)
"""
DDM_OWN = "ddm-own-v2"
IDMT = "IDMT-SMT-DRUMS-V2"
ENST = "ENST-drums-public"
E_GMD = "e-gmd-v1.0.0"
DRUM_KIT = "drum-kit-sound"


"""
-- model type
"""
# -- segment & classify 방식
METHOD_CLASSIFY = "classify"

# -- separate & detect 방식
METHOD_DETECT = "detect"

# -- 박자 인식 모델
METHOD_RHYTHM = "rhythm"


"""
-- related audio
"""
# -- 일정한 시간 간격으로 음압을 측정하는 주파수, 44100Hz (단위 Hz)
SAMPLE_RATE = 44100

# -- 오디오 자를 시, onset 기준 왼쪽, 오른쪽으로 몇 초 자를지 (단위: sec)
ONSET_DURATION_LEFT = 0.03
ONSET_DURATION_RIGHT = 0.5

# -- onset offset: int (onset position 양 옆으로 몇 개씩 붙일지)
ONSET_OFFSET = 1

# -- chunk time - feature 추출 시
CHUNK_LENGTH = 12

# -- chunk length - model에 넣기 전 dataset 가공 시
CHUNK_TIME_LENGTH = 30

# -- unit (1 sec)
MILLISECOND = 1000000

# -- classify method feature duration
CLASSIFY_DURATION = ONSET_DURATION_RIGHT + 0.1


"""
-- drum mapping

-- 파일 이름 형식
-- per_drum : CC_04_9949.wav
-- pattern : P1_08_0001.wav
"""
CODE2DRUM = {0: "CC", 1: "OH", 2: "CH", 3: "TT", 4: "SD", 5: "KK"}
# -- {'CC':0, 'OH':1, ...}
DRUM2CODE = {v: k for k, v in CODE2DRUM.items()}
# -- {'CC':[1,0,0,0,0,0], 'OH':[0,1,0,0,0,0], ...}
ONEHOT_DRUM2CODE = {}
for code, index in DRUM2CODE.items():
    drum_mapping = [0] * len(DRUM2CODE)
    drum_mapping[index] = 1
    ONEHOT_DRUM2CODE[code] = drum_mapping

PATTERN = {
    "CH": ONEHOT_DRUM2CODE["CH"],
    "SD": ONEHOT_DRUM2CODE["SD"],
    "CH_KK": [0, 0, 1, 0, 0, 1],
    "CH_SD": [0, 0, 1, 0, 1, 0],
}

P_HH_KK = PATTERN["CH_KK"]
P_SD = PATTERN["SD"]
P_HH = PATTERN["CH"]
P_HH_SD = PATTERN["CH_SD"]

P1_2CODE = [P_HH_KK, P_HH, P_HH_SD, P_HH, P_HH_KK, P_HH_KK, P_HH_SD, P_HH]
P2_2CODE = [
    P_HH_KK,
    P_HH,
    P_HH,
    P_HH,
    P_SD,
    P_HH,
    P_HH,
    P_HH,
    P_HH_KK,
    P_HH,
    P_HH,
    P_HH,
    P_SD,
    P_HH,
    P_HH,
    P_HH,
]
PATTERN2CODE = {"P1": P1_2CODE, "P2": P2_2CODE}


"""
-- 우리 데이터랑 연관된 상수
"""
# -- dir name
PATTERN_DIR = "pattern"
PER_DRUM_DIR = "per-drum"


"""
-- data origin별로 사용하는 data path와 관련된 상수
"""
DATA_IDMT = (
    "MIX",
    "RealDrum",
    "WaveDrum",
)
DATA_IDMT_NOT = (
    "train",
    "TechnoDrum",
)
DATA_ENST = (
    "hits_snare-drum_sticks",
    "hits_snare-drum_mallets",
    "hits_medium-tom_sticks",
    "hits_bass-drum_pedal",
    "hits_pedal-hi-hat-close_pedal",
    "hits_pedal-hi-hat-open_pedal",
    "sticks",
)
DATA_ENST_NOT = ("accompaniment",)  # ENST dataset에서 제외할 데이터
DATA_DDM_OWN = (
    "per-drum/CC",
    "per-drum/HH",
    "per-drum/MT",
    "per-drum/SD",
    "per-drum/KK",
    "pattern/P1",
    "pattern/P2",
)

DATA_ALL = DATA_IDMT + DATA_ENST + DATA_DDM_OWN + (DRUM_KIT,) + (E_GMD,)

# -------------------------------------------------------------------------------------
"""
-- Mapping drum_type
DRUM_TYPES에 추가하기

=> DRUM_MAP을 접근해서 사용 {"sd": "SD", "mt": "TT", "bd": "KK", "chh": "CH", "ohh": "OH"}
"""
DRUM_TYPES = {
    "CC": [
        49,  # crash cymbal 1
        57,  # crash cymbal 2
        52,  # china cymbal
        55,  # splash cymbal
        51,  # ride cymbal
        59,  # ride cymbal 2
        "CC",  # crash (ddm-own)
    ],  # crash
    "OH": [
        "ohh",
        46,  # hi-hat open
        "overheads",  # drum kit data
    ],  # hi-hat open
    "CH": [
        "chh",
        42,  # hi-hat cloased
        "HH",  # closed hi-hat (ddm-own)
    ],  # hi-hat closed
    "TT": [
        "mt",
        45,  # mid tom
        47,  # mid tom
        48,  # high-mid tom
        50,  # high tom
        "toms",  # tom (drum kit data)
    ],  # tom
    "SD": [
        "sd",
        38,  # snare drum
        40,  # electric snare drum
        "snare",  # snare drum (drum kit data)
        "SD",  # snare (ddm-own)
    ],  # snare
    "KK": [
        "bd",
        35,  # bass drum
        36,  # kick drum
        "kick",  # kick (drum kit data)
        "KD",  # kick (idmt)
        "KK",  # kick (ddm-own)
    ],  # kick
}
DRUM_MAP = {}
# Iterate over the DRUM_TYPES
for drum_type, values in DRUM_TYPES.items():
    # Iterate over the values for each drum_type
    for value in values:
        # Add the mapping to the new dictionary
        DRUM_MAP[value] = drum_type


"""
-- classify 방법에서의 분류 라벨
"""
CLASSIFY_DETECT_TYPES = {
    "OH": ["CC", "OH", "CH"],
    # : [
    #     "CH",
    # ],
    "TT": [
        "TT",
    ],
    "SD": [
        "SD",
    ],
    "KK": [
        "KK",
    ],
}
CLASSIFY_MAP = {}
# Iterate over the DRUM_TYPES
for drum_type, values in CLASSIFY_DETECT_TYPES.items():
    # Iterate over the values for each drum_type
    for value in values:
        # Add the mapping to the new dictionary
        CLASSIFY_MAP[value] = drum_type
"""
-- {0: "OH", 1: "CH", ...}
"""
CLASSIFY_CODE2DRUM = {i: k for i, k in enumerate(CLASSIFY_DETECT_TYPES.keys())}
"""
-- {"OH": 0, "CH": 1, ...}
"""
CLASSIFY_DRUM2CODE = {v: k for k, v in CLASSIFY_CODE2DRUM.items()}
"""
-- classify 방법에서 불가능한 라벨 값 (십진수)
"""
CLASSIFY_IMPOSSIBLE_LABEL = {14, 15, 22, 23, 26, 27, 28, 29, 30, 31}


"""
-- feature type
"""
MFCC = "mfcc"
STFT = "stft"
MEL_SPECTROGRAM = "mel-spectrogram"


"""
-- feature parameter
"""
FEATURE_DTYPE = "float16"
FEATURE_PARAM_BASIC = {
    "n_fft": 2048,
    "win_length": 1024,
    "hop_length": 441,
    "n_classes": len(CODE2DRUM),
}
FEATURE_PARAM = {
    METHOD_CLASSIFY: {
        MFCC: {
            **FEATURE_PARAM_BASIC,
            "n_mfcc": 40,
            "n_channels": 1,
            "n_classes": len(CLASSIFY_CODE2DRUM),
        },
        STFT: {
            **FEATURE_PARAM_BASIC,
            "n_classes": len(CLASSIFY_CODE2DRUM),
        },
        MEL_SPECTROGRAM: {
            **FEATURE_PARAM_BASIC,
            "n_mels": 128,  # -- number of mel bands
            "fmin": 27.5,
            "fmax": 16000,
            "n_classes": len(CLASSIFY_CODE2DRUM),
        },
    },
    METHOD_DETECT: {
        MFCC: {
            **FEATURE_PARAM_BASIC,
            "n_mfcc": 40,
            "n_channels": 1,
        },
        STFT: {**FEATURE_PARAM_BASIC},
        MEL_SPECTROGRAM: {
            **FEATURE_PARAM_BASIC,
            "n_mels": 128,  # -- number of mel bands
            "fmin": 27.5,
            "fmax": 16000,
            "n_classes": len(CLASSIFY_CODE2DRUM),
        },
    },
    METHOD_RHYTHM: {
        MFCC: {
            **FEATURE_PARAM_BASIC,
            "n_classes": 1,
            "n_mfcc": 40,
            "n_channels": 1,
        },
        STFT: {
            **FEATURE_PARAM_BASIC,
            "n_classes": 1,
        },
        # -- adt 논문 참고 파라미터
        MEL_SPECTROGRAM: {
            **FEATURE_PARAM_BASIC,
            "n_classes": 1,
            "n_mels": 128,  # -- number of mel bands
            "fmin": 27.5,
            "fmax": 16000,
        },
    },
}


"""
-- feature extension 
"""
CSV = "csv"
PKL = "pkl"
