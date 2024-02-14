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
ONSET_OFFSET = 0

# -- chunk time
CHUNK_LENGTH = 12

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
CODE2DRUM = {0: "HH", 1: "ST", 2: "SD", 3: "KK"}
# -- {'HH':0, 'MT':1, ...}
DRUM2CODE = {v: k for k, v in CODE2DRUM.items()}
# -- {'HH':[1,0,0,0], 'MT':[0,1,0,0], ...}
ONEHOT_DRUM2CODE = {}
for code, index in DRUM2CODE.items():
    drum_mapping = [0] * len(DRUM2CODE)
    drum_mapping[index] = 1
    ONEHOT_DRUM2CODE[code] = drum_mapping

PATTERN = {
    "HH": ONEHOT_DRUM2CODE["HH"],
    "SD": ONEHOT_DRUM2CODE["SD"],
    "HH_KK": [1, 0, 0, 1],
    "HH_SD": [1, 0, 1, 0],
}

P_HH_KK = PATTERN["HH_KK"]
P_SD = PATTERN["SD"]
P_HH = PATTERN["HH"]
P_HH_SD = PATTERN["HH_SD"]

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
        },
        STFT: {**FEATURE_PARAM_BASIC},
        MEL_SPECTROGRAM: {
            **FEATURE_PARAM_BASIC,
            "n_mels": 128,  # -- number of mel bands
            "fmin": 27.5,
            "fmax": 16000,
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


"""
-- 우리 데이터랑 연관된 상수
"""
# -- dir name
PATTERN_DIR = "pattern"
PER_DRUM_DIR = "per-drum"


"""
-- classify 방법에서 사용하는 data path와 관련된 상수
"""
DATA_IDMT = ("WaveDrum02", "MIX")
DATA_IDMT_NOT = ("train",)
DATA_ENST = (
    "hits_snare-drum_sticks",
    "hits_snare-drum_mallets",
    "hits_medium-tom_sticks",
    "hits_bass-drum_pedal",
    "hits_pedal-hi-hat-close_pedal",
    "hits_pedal-hi-hat-open_pedal",
)
DATA_ENST_NOT = ("accompaniment",)  # ENST dataset에서 제외할 데이터
DATA_DDM_OWN = (
    "per-drum/HH",
    "per-drum/MT",
    "per-drum/SD",
    "per-drum/KK",
    "pattern/P1",
    "pattern/P2",
)

DATA_ALL = DATA_IDMT + DATA_ENST + DATA_DDM_OWN + (DRUM_KIT,) + (E_GMD,)

CLASSIFY_DRUM = {
    0: (
        "HH",
        "hi-hat",
    ),
    1: ("tom",),
    2: (
        "SD",
        "snare",
    ),
    3: (
        "KD",
        "KK",
        "bass",
    ),
}

# -------------------------------------------------------------------------------------
"""
-- Mapping drum_type
DRUM_TYPES에 추가하기

=> DRUM_MAP을 접근해서 사용 {"sd": "SD", "mt": "ST", "bd": "KK", "chh": "HH", "ohh": "HH"}
"""
DRUM_TYPES = {
    "HH": [
        "chh",
        "ohh",
        42,
        46,
    ],
    "ST": [
        "mt",
        48,
        50,
    ],
    "SD": [
        "sd",
        38,
        40,
    ],
    "KK": [
        "bd",
        35,
        36,
    ],
}
DRUM_MAP = {}
# Iterate over the DRUM_TYPES
for drum_type, values in DRUM_TYPES.items():
    # Iterate over the values for each drum_type
    for value in values:
        # Add the mapping to the new dictionary
        DRUM_MAP[value] = drum_type
