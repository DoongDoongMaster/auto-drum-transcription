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
-- split data type
"""
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

"""
-- data origin (데이터 출처)
"""
DDM_OWN = "ddm-own-v2"
IDMT = "IDMT-SMT-DRUMS-V2"
ENST = "ENST-drums-public-clean"
E_GMD = "e-gmd-v1.0.0"
DRUM_KIT = "drum-kit-sound"


"""
-- 우리 데이터랑 연관된 상수
"""
# -- dir name
PATTERN_DIR = "pattern"
PER_DRUM_DIR = "per-drum"


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

# -- classify method feature duration
CLASSIFY_DURATION = ONSET_DURATION_RIGHT + 0.1

# -- 동시에 친 오디오 구분 초 (단위: sec)
CLASSIFY_SAME_TIME = 0.035

# -- 너무 짧게 잘린 데이터 버리는 단위 (단위: sec)
CLASSIFY_SHORT_TIME = 0.16

# -- onset offset: int (onset position 양 옆으로 몇 개씩 붙일지)
ONSET_OFFSET = 1


# -- chunk time - feature 추출 시
CHUNK_LENGTH = 12

# -- chunk length - model에 넣기 전 dataset 가공 시
CHUNK_TIME_LENGTH = 30

# -- unit (1 sec)
MILLISECOND = 1000000


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
DATA_ENST_NOT = (
    "accompaniment",
    "cowbell",
    "brushes",
    "no-snare",
    "mallets",
    "rods",
    "cross-sticks",
    "phrase_reggae",
    "salsa",
    # -- drummer1
    "096_solo_latin_hands",  # sd- (no-snare)
    "108_minus-one_rock-60s_sticks",  # cb (cow bell)
    # -- drummer2
    "108_solo_toms_sticks",  # sd-
    "121_minus-one_charleston_sticks",  # cs (cross sticks)
    "124_minus-one_bossa_sticks",  # cs
    "134_MIDI-minus-one_country-120_sticks",  # cs
    "144_MIDI-minus-one_rock-113_sticks",  # cs
    "150_MIDI-minus-one_soul-98_sticks",  # cs
    # -- drummer3
    "067_phrase_afro_simple_slow_sticks",  # cs
    "068_phrase_afro_simple_medium_sticks",  # cs
    "090_phrase_shuffle-blues_complex_fast_sticks",  # sticks
    "111_phrase_oriental_simple_fast_sticks",  # cs
    "114_phrase_oriental_complex_fast_sticks",  # cs
    "115_phrase_cha-cha_simple_slow_sticks",  # cs
    "116_phrase_cha-cha_complex_slow_sticks",  # cb
    "134_minus-one_bossa_sticks",  # cs
    "140_MIDI-minus-one_bigband_sticks",  # cs
    "160_MIDI-minus-one_soul-98_sticks",  # cs
)  # ENST dataset에서 제외할 데이터
DATA_DDM_OWN = (
    "per-drum/CC",
    "per-drum/HH",
    "per-drum/MT",
    "per-drum/SD",
    "per-drum/KK",
    "pattern/P1",
    "pattern/P2",
)
DATA_E_GMD_NOT = (
    "drummer1/session2/66_punk_144_fill_4-4",  # 싱크 안 맞음
    "drummer7/session3/25_hiphop_67_fill_4-4",  # wav 파일 비어있음
    "drummer7/session3/109_rock_95_beat_4-4",  # 싱크 안 맞음
    "drummer7/session2/81_country_78_fill_4-4",  # 싱크 안 맞음
    "drummer3/session2/2_rock_100_beat_4-4",  # 싱크 안 맞음
    "drummer7/session3/146_soul_105_fill_4-4",  # 싱크 안 맞음
    "drummer7/session1/15_jazz_112_beat_4-4",  # 싱크 안 맞음
    "drummer7/session3/149_soul_105_fill_4-4",  # 싱크 안 맞음
    "drummer7/session3/63_funk_112_fill_4-4",  # 싱크 안 맞음
    "drummer3/session1/9_rock_105_beat_4-4",  # 싱크 안 맞음
    "drummer7/session3/25_hiphop_67_fill_4-4",  # 싱크 안 맞음
    "drummer7/session3/156_soul_98_fill_4-4",  # 싱크 안 맞음
    "drummer1/session1/5_jazz-funk_116_beat_4-4",  # 싱크 안 맞음
)
DATA_E_GMD_NOT = DATA_E_GMD_NOT + tuple(
    f"_{i}.wav" for i in range(2, 59)
)  # acustic kit 만 사용

DATA_ALL = DATA_IDMT + DATA_DDM_OWN + (DRUM_KIT,) + (E_GMD,) + (ENST,)

# -------------------------------------------------------------------------------------

"""
train/test split info
"""
# 'wet_mix' 폴더 내에 'minus-one'이 포함되어 있는지 확인
DATA_ENST_TEST = {"directory": "wet_mix", "test": "_minus-one_"}
E_GMD_INFO = f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}/info.csv"

# -------------------------------------------------------------------------------------

"""
-- Mapping drum_type
DRUM_TYPES에 추가하기

=> DRUM_MAP을 접근해서 사용 {"sd": "SD", "mt": "TT", "bd": "KK", "chh": "CH", "ohh": "OH"}
"""
DRUM_TYPES = {
    "CC": [
        27,  # -- china 1
        28,  # -- cymbal 1
        30,  # -- cymbal 3
        31,  # -- cymbal 4
        32,  # -- cymbal 5
        49,  # crash cymbal 1
        57,  # crash cymbal 2
        52,  # china cymbal
        55,  # splash cymbal
        51,  # ride cymbal
        59,  # ride cymbal 2
        53,  # ride bell
        "CC",  # crash (ddm-own)
        "c1",  # crash cymbal 1 (enst/drummer1,2)
        "cr1",  # crash cymbal 1 (enst/drummer2)
        "cr2",  # crash cymbal 1 (enst/drummer3)
        "cr5",  # crash cymbal 2 (enst/drummer3)
        "rc3",  # ride cymbal 1 (enst/drummer2)
        "rc2",  # ride cymbal 2 (enst/drummer1)
        "rc4",  # ride cymbal 2 (enst/drummer2)
        "c4",  # ride cymbal 2 (enst/drummer3)
        "ch5",  # china ride cymbal (enst/drummer2)
        "ch1",  # china ride cymbal (enst/drummer3)
        "spl2",  # splash cymbal (enst/drummer2)
    ],  # crash
    "OH": [
        23,  # -- open pedal
        24,  # -- open 1
        25,  # -- open 2
        26,  # -- open 3
        "ohh",
        46,  # hi-hat open
        "overheads",  # drum kit data
    ],  # hi-hat open
    "CH": [
        21,  # -- closed pedal
        22,  # -- closed Edge
        "chh",
        42,  # hi-hat cloased
        44,  # hi-hat pedal
        "HH",  # closed hi-hat (ddm-own)
    ],  # hi-hat closed
    "TT": [
        "mt",
        41,  # -- low tom 2
        43,  # -- low tom 1
        45,  # mid tom
        47,  # mid tom
        48,  # high-mid tom
        50,  # high tom
        58,  # -- vibra slap
        "toms",  # tom (drum kit data)
        "ltr",  # low-tom, hit on the rim (enst/drummer1)
        "lmt",  # mid-tom-2 (enst/drummer3)
        "lt",  # low-tom (enst)
        "lft",  # low-tom-2 (enst/drummer3)
    ],  # tom
    "SD": [
        "sd",
        37,  # rimshot
        38,  # snare drum
        40,  # electric snare drum
        "snare",  # snare drum (drum kit data)
        "SD",  # snare (ddm-own)
        "rs",  # rim shot (enst)
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

CODE2DRUM = {i: k for i, k in enumerate(DRUM_TYPES.keys())}
# -- {'CC':0, 'OH':1, ...}
DRUM2CODE = {v: k for k, v in CODE2DRUM.items()}


"""
-- labeling type
"""
LABEL_REF = "LABEL_REF"
LABEL_DDM = "LABEL_DDM"
LABEL_REF_COLUMN = []  # -- HH-LABEL_REF, ST-LABEL_REF, SD-LABEL_REF, KK-LABEL_REF
LABEL_DDM_COLUMN = []  # -- HH-LABEL_DDM, ST-LABEL_DDM, SD-LABEL_DDM, KK-LABEL_DDM
for _, drum_code in CODE2DRUM.items():
    LABEL_REF_COLUMN.append(f"{drum_code}-{LABEL_REF}")
    LABEL_DDM_COLUMN.append(f"{drum_code}-{LABEL_DDM}")
LABEL_TYPE = {
    LABEL_REF: {
        "labeled": "[0.5_1.0_0.5]",
        "offset": {"l": [0.5], "r": [0.5]},
        "column": LABEL_REF_COLUMN,
    },
    LABEL_DDM: {
        "labeled": "[1.0_1.0_0.5]",
        "offset": {"l": [], "r": [1.0, 0.5]},
        "column": LABEL_DDM_COLUMN,
    },
}
LABEL_COLUMN = []
for label, _ in LABEL_TYPE.items():
    for _, drum_code in CODE2DRUM.items():
        LABEL_COLUMN.append(f"{drum_code}-{label}")
LABEL_INIT_DATA = {
    IDMT: {TRAIN: [], TEST: []},
    ENST: {TRAIN: [], TEST: []},
    E_GMD: {TRAIN: [], VALIDATION: [], TEST: []},
}
"""
-- classify 방법에서의 분류 라벨
"""
CLASSIFY_TYPES = {
    "OH": [
        "CC",
        "OH",
    ],
    "CH": [
        "CH",
    ],
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
for drum_type, values in CLASSIFY_TYPES.items():
    # Iterate over the values for each drum_type
    for value in values:
        # Add the mapping to the new dictionary
        CLASSIFY_MAP[value] = drum_type
"""
-- {0: "OH", 1: "CH", ...}
"""
CLASSIFY_CODE2DRUM = {i: k for i, k in enumerate(CLASSIFY_TYPES.keys())}
"""
-- classify 방법에서 불가능한 라벨 값 (십진수)
"""
CLASSIFY_IMPOSSIBLE_LABEL = (
    {14, 15, 22, 23, 26, 27, 28, 29, 30, 31} if len(CLASSIFY_TYPES) == 5 else {0}
)


"""
-- detect 방법에서의 분류 라벨
"""
DETECT_TYPES = {
    "OH": ["CC", "OH", "CH"],
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
DETECT_MAP = {}
for drum_type, values in DETECT_TYPES.items():
    for value in values:
        DETECT_MAP[value] = drum_type
"""
-- {0: "OH", 1: "TT", ...}
"""
DETECT_CODE2DRUM = {i: k for i, k in enumerate(DETECT_TYPES.keys())}


"""
-- drum mapping

-- 파일 이름 형식
-- per_drum : CC_04_9949.wav
-- pattern : P1_08_0001.wav
"""
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


# ------------------------------------------------------------------------------------

"""
-- feature type
"""
MFCC = "mfcc"
STFT = "stft"
MEL_SPECTROGRAM = "mel-spectrogram"


"""
-- feature parameter
"""
FEATURE_DTYPE_16 = "float16"
FEATURE_DTYPE_32 = "float32"
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
            "n_classes": len(DETECT_CODE2DRUM),
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
-- classify feature 추출 시, 필요한 상수 (단위: frame)
"""
CLASSIFY_DURATION_FRAME = round(
    CLASSIFY_DURATION * (SAMPLE_RATE // FEATURE_PARAM_BASIC["hop_length"])
)
CLASSIFY_SAME_TIME_FRAME = round(
    CLASSIFY_SAME_TIME * (SAMPLE_RATE // FEATURE_PARAM_BASIC["hop_length"])
)
CLASSIFY_SHORT_TIME_FRAME = round(
    CLASSIFY_SHORT_TIME * (SAMPLE_RATE // FEATURE_PARAM_BASIC["hop_length"])
)


"""
-- feature extension 
"""
CSV = "csv"
PKL = "pkl"
