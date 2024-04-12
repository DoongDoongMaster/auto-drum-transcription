import os
import re

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
ENST_PUB = "ENST-drums-public"
E_GMD = "e-gmd-v1.0.0"
DRUM_KIT = "drum-kit-sound"
MDB = "MDBDrums"
MDB_LABEL_TYPE = "subclass"


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
-- model saved type
"""
MODEL_SAVED_H5 = "h5"
MODEL_SAVED_PB = "pb"
MODEL_DIR = "models"


"""
-- served model
"""
SERVED_MODEL_DIR = "served-models"  # 서빙 모델 최상위 폴더
SERVED_MODEL_TYPE = (
    METHOD_CLASSIFY,
    METHOD_DETECT,
    METHOD_RHYTHM,
)
REDIS_AI_HOST = "localhost"
REDIS_AI_PORT = 6379


"""
-- related audio
"""
# -- 일정한 시간 간격으로 음압을 측정하는 주파수, 44100Hz (단위 Hz)
SAMPLE_RATE = 44100

# -- 오디오 자를 시, onset 기준 왼쪽, 오른쪽으로 몇 초 자를지 (단위: sec)
ONSET_DURATION_LEFT = 0.025
ONSET_DURATION_RIGHT_MINUS = 0
ONSET_DURATION_RIGHT = 0.125

# -- classify method feature duration
CLASSIFY_DURATION = 0.16

# -- 동시에 친 오디오 구분 초 (단위: sec)
CLASSIFY_SAME_TIME = 0.16

# -- 너무 짧게 잘린 데이터 버리는 단위 (단위: sec)
CLASSIFY_SHORT_TIME = 0.135

# -- onset offset: int (onset position 양 옆으로 몇 개씩 붙일지)
ONSET_OFFSET = 1


# -- chunk time - feature 추출 시
CHUNK_LENGTH = 12

# -- chunk length - model에 넣기 전 dataset 가공 시
CHUNK_TIME_LENGTH = 1200

# -- unit (1 sec)
MILLISECOND = 1000000


"""
-- data origin별로 사용하는 data path와 관련된 상수
"""
DATA_IDMT = (
    "RealDrum",
    "WaveDrum",
)
DATA_IDMT_NOT = ("TechnoDrum",)
DATA_ENST_NOT = (
    "accompaniment/",
    "brushes",
    "mallets",
    "rods",
    "002_hits_snare-drum-no-snare_sticks_x5",
    "011_hits_cowbell_sticks_x5",
    "020_hits_cowbell_brushes_x5",
    "022_hits_snare-drum-no-snare_mallets_x5",
    "028_hits_cowbell_mallets_x4",
    "029_hits_cross-sticks_sticks_x4",
    "049_phrase_afro_simple_medium_mallets",
    "050_phrase_afro_simple_fast_mallets",
    "051_phrase_afro_complex_slow_mallets",
    "052_phrase_afro_complex_medium_mallets",
    "053_phrase_afro_complex_fast_mallets",
    "060_phrase_salsa_simple_slow_sticks",
    "061_phrase_salsa_simple_medium_sticks",
    "062_phrase_salsa_simple_fast_sticks",
    "063_phrase_salsa_complex_slow_sticks",
    "065_phrase_salsa_complex_fast_sticks",
    "078_phrase_reggae_simple_slow_sticks",
    "079_phrase_reggae_simple_medium_sticks",
    "080_phrase_reggae_simple_fast_sticks",
    "081_phrase_reggae_complex_slow_sticks",
    "082_phrase_reggae_complex_medium_sticks",
    "083_phrase_reggae-ska_complex_fast_sticks",
    "096_solo_latin_hands",
    "098_solo_afro_mallets",
    "101_solo_jazz-rock_rods",
    "102_solo_salsa_sticks",
    "107_minus-one_salsa_sticks",
    "108_minus-one_rock-60s_sticks",
    "112_minus-one_funk_rods",
    "114_minus-one_celtic-rock_brushes",
    "115_minus-one_bossa_brushes",
    "013_hits_cowbell_sticks_x7",
    "014_hits_snare-drum-shuffle_brushes_x7",
    "024_hits_cowbell_brushes_x8",
    "028_hits_snare-drum-no-snare_mallets_x7",
    "108_solo_toms_sticks",
    "115_minus-one_salsa_sticks",
    "121_minus-one_charleston_sticks",
    "124_minus-one_bossa_sticks",
    "134_MIDI-minus-one_country-120_sticks",
    "144_MIDI-minus-one_rock-113_sticks",
    "150_MIDI-minus-one_soul-98_sticks",
    "041_hits_snare-drum-no-snare_mallets_x5",
    "048_hits_cross-sticks_sticks_x5",
    "067_phrase_afro_simple_slow_sticks",
    "068_phrase_afro_simple_medium_sticks",
    "079_phrase_salsa_simple_slow_sticks",
    "080_phrase_salsa_simple_medium_sticks",
    "081_phrase_salsa_simple_medium_sticks",
    "082_phrase_salsa_complex_slow_sticks",
    "090_phrase_shuffle-blues_complex_fast_sticks",
    "091_phrase_reggae_simple_slow_sticks",
    "092_phrase_reggae_simple_medium_sticks",
    "093_phrase_reggae_simple_fast_sticks",
    "094_phrase_reggae_complex_slow_sticks",
    "095_phrase_reggae_complex_medium_sticks",
    "096_phrase_reggae_complex_fast_sticks",
    "111_phrase_oriental_simple_fast_sticks",
    "114_phrase_oriental_complex_fast_sticks",
    "115_phrase_cha-cha_simple_slow_sticks",
    "116_phrase_cha-cha_complex_slow_sticks",
    "119_solo_toms_mallets",
    "126_minus-one_salsa_sticks",
    "134_minus-one_bossa_sticks",
    "140_MIDI-minus-one_bigband_sticks",
    "160_MIDI-minus-one_soul-98_sticks",
    # -- hi-hat data만 사용
    # "dry_mix/",
    # "kick/",
    # "overhead_L/",
    # "overhead_R/",
    # "snare/",
    # "tom_1/",
    # "tom_2/",
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
DATA_MDB = ("MDBDrums/audio/drum_only/",)
DATA_MDB_NOT = (
    "MusicDelta_Reggae_",  # no snare,
    "MusicDelta_SwingJazz_",  # brush,
    "MusicDelta_CoolJazz_",  # brush
    "MusicDelta_Beatles_",  # tambourine
)
DATA_E_GMD_NOT = DATA_E_GMD_NOT + tuple(
    f"_{i}.wav" for i in range(2, 59)
)  # acustic kit 만 사용

DATA_ALL = (
    DATA_IDMT + DATA_DDM_OWN + DATA_MDB + (DRUM_KIT,) + (E_GMD,) + (ENST,) + (ENST_PUB,)
)
# -------------------------------------------------------------------------------------

"""
train/test split info
"""
# 'wet_mix' 폴더 내에 'minus-one'이 포함되어 있는지 확인
DATA_ENST_TEST = {"directory": "wet_mix", "test": "_minus-one_"}
E_GMD_INFO = f"{ROOT_PATH}/{RAW_PATH}/{E_GMD}/{E_GMD}.csv"
MDB_INFO = f"{ROOT_PATH}/{RAW_PATH}/{MDB}/MIREX2017.md"
MDB_TRAIN_SET = []
if os.path.exists(MDB_INFO):  # 파일 존재 여부 확인
    with open(MDB_INFO, "r") as f:
        content = f.read()

        # Extracting training set
        train_matches = re.findall(
            r"### Training Set\n([\s\S]*?)\n### Test Set", content
        )
        if train_matches:
            train_tracks = train_matches[0].strip().split("\n")
            MDB_TRAIN_SET = (track.strip() for track in train_tracks)
    MDB_TRAIN_SET = list(MDB_TRAIN_SET)


# -------------------------------------------------------------------------------------


"""
-- drum name
"""
CC = "CC"  # crush symbal
RD = "RD"  # ride cymbal & ride bell
OH = "OH"  # open hi-hat
CH = "CH"  # closed hi-hat
TT = "TT"  # tom
SD = "SD"  # snare
RS = "RS"  # rimshot
KK = "KK"  # kick


"""
-- Mapping drum_type
DRUM_TYPES에 추가하기

=> DRUM_MAP을 접근해서 사용 {"sd": "SD", "mt": "TT", "bd": "KK", "chh": "CH", "ohh": "OH"}
"""
DRUM_TYPES = {
    CC: [
        27,  # -- china 1
        28,  # -- cymbal 1
        30,  # -- cymbal 3
        31,  # -- cymbal 4
        32,  # -- cymbal 5
        49,  # crash cymbal 1
        57,  # crash cymbal 2
        52,  # china cymbal
        55,  # splash cymbal
        "CC",  # crash (ddm-own)
        "c1",  # crash cymbal 1 (enst/drummer1,2)
        "cr1",  # crash cymbal 1 (enst/drummer2)
        "cr2",  # crash cymbal 1 (enst/drummer3)
        "cr5",  # crash cymbal 2 (enst/drummer3)
        "c4",  # ride cymbal 2 (enst/drummer3)
        "ch5",  # china ride cymbal (enst/drummer2)
        "ch1",  # china ride cymbal (enst/drummer3)
        "spl2",  # splash cymbal (enst/drummer2)
        "CRC",  # crash cymbal (MDB)
        "CHC",  # china cymbal (MDB)
        "SPC",  # splash cymbal (MDB)
    ],  # crash
    RD: [
        51,  # ride cymbal
        59,  # ride cymbal 2
        53,  # ride bell
        "rc3",  # ride cymbal 1 (enst/drummer2)
        "rc2",  # ride cymbal 2 (enst/drummer1)
        "rc4",  # ride cymbal 2 (enst/drummer2)
        "RDC",  # ride cymbal (MDB)
        "RDB",  # ride cymbal bell (MDB)
    ],  # ride
    OH: [
        23,  # -- open pedal
        24,  # -- open 1
        25,  # -- open 2
        26,  # -- open 3
        "ohh",
        46,  # hi-hat open
        "overheads",  # drum kit data
        "OHH",  # open hi-hat (MDB)
    ],  # hi-hat open
    CH: [
        21,  # -- closed pedal
        22,  # -- closed Edge
        "chh",
        42,  # hi-hat cloased
        44,  # hi-hat pedal
        "HH",  # closed hi-hat (ddm-own)
        "CHH",  # closed hi-hat (MDB)
        "PHH",  # pedal hi-hat (MDB)
        "TMB",  # tambourine (MDB)
    ],  # hi-hat closed
    TT: [
        "mt",
        41,  # -- low tom 2
        43,  # -- low tom 1
        45,  # mid tom
        47,  # mid tom
        48,  # high-mid tom
        50,  # high tom
        58,  # -- vibra slap
        "toms",  # tom (drum kit data)
        "lmt",  # mid-tom-2 (enst/drummer3)
        "lt",  # low-tom (enst)
        "lft",  # low-tom-2 (enst/drummer3)
        "TT",  # tom (MDB)
        "HIT",  # high tom (MDB)
        "MHT",  # high-mid tom (MDB)
        "HFT",  # high floor tom (MDB)
        "LFT",  # low floor tom (MDB)
    ],  # tom
    SD: [
        "sd",
        38,  # snare drum
        40,  # electric snare drum
        "snare",  # snare drum (drum kit data)
        "SD",  # snare (ddm-own) & (MDB)
        "SDD",  # snare: drag (MDB)
        "SDF",  # snare: flam (MDB)
        "SDG",  # snare: gohst note (MDB)
        "SDB",  # snare: brush (MDB)
        "SDNS",  # snare: no snare (MDB)
    ],  # snare
    RS: [
        37,  # rimshot
        "ltr",  # low-tom, hit on the rim (enst/drummer1)
        "rs",  # rim shot (enst)
        "SST",  # side stick (MDB)
    ],  # rimshot
    KK: [
        "bd",
        35,  # bass drum
        36,  # kick drum
        "kick",  # kick (drum kit data)
        "KD",  # kick (idmt) & (MDB)
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
LABEL_DDM_1 = "LABEL_DDM_1"
LABEL_TYPE = {
    LABEL_REF: {
        "labeled": "[0.5_1.0_0.5]",
        "offset": {"l": [0.5], "r": [0.5]},
        # -- HH-LABEL_REF, ST-LABEL_REF, SD-LABEL_REF, KK-LABEL_REF
        "column": [f"{drum_code}-{LABEL_REF}" for _, drum_code in CODE2DRUM.items()],
    },
    LABEL_DDM: {
        "labeled": "[1.0_1.0_0.5]",
        "offset": {"l": [], "r": [1.0, 0.5]},
        # -- HH-LABEL_DDM, ST-LABEL_DDM, SD-LABEL_DDM, KK-LABEL_DDM
        "column": [f"{drum_code}-{LABEL_DDM}" for _, drum_code in CODE2DRUM.items()],
    },
    LABEL_DDM_1: {
        "labeled": "[1.0_0.5_0.5]",
        "offset": {"l": [], "r": [0.5, 0.5]},
        # -- HH-LABEL_DDM_1, ST-LABEL_DDM_1, SD-LABEL_DDM_1, KK-LABEL_DDM_1
        "column": [f"{drum_code}-{LABEL_DDM_1}" for _, drum_code in CODE2DRUM.items()],
    },
}
LABEL_COLUMN = []
for label, data in LABEL_TYPE.items():
    LABEL_COLUMN += data["column"]

LABEL_INIT_DATA = {
    IDMT: {
        TRAIN: [],
        TEST: [],
    },
    ENST: {
        TRAIN: [],
        TEST: [],
    },
    ENST_PUB: {
        TRAIN: [],
        TEST: [],
    },
    E_GMD: {
        TRAIN: [],
        VALIDATION: [],
        TEST: [],
    },
    MDB: {
        TRAIN: [],
        TEST: [],
    },
    DRUM_KIT: {
        TRAIN: [],
    },
}


"""
-- 라벨 개수에 따른 drum_types
"""
DRUM_TYPES_3 = {
    OH: [
        CC,
        OH,
        CH,
        RD,
    ],
    SD: [
        TT,
        SD,
        RS,
    ],
    KK: [
        KK,
    ],
}
DRUM_TYPES_4 = {
    OH: [
        CC,
        OH,
        CH,
        RD,
    ],
    TT: [
        TT,
    ],
    SD: [
        SD,
        RS,
    ],
    KK: [
        KK,
    ],
}


"""
-- classify 방법에서의 분류 라벨
"""
CLASSIFY_TYPES = DRUM_TYPES_3
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
-- detect 방법에서의 분류 라벨
"""
DETECT_TYPES = DRUM_TYPES_3
DETECT_MAP = {}
for drum_type, values in DETECT_TYPES.items():
    for value in values:
        DETECT_MAP[value] = drum_type
"""
-- {0: "OH", 1: "TT", ...}
"""
DETECT_CODE2DRUM = {i: k for i, k in enumerate(DETECT_TYPES.keys())}


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
    "win_length": 2048,
    "hop_length": 512,
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


"""
-- stored model name
"""
# ------------ 0329 -------------
SERVED_MODEL_CLASSIFY_LSTM = "classify_mel-spectrogram_2024-03-15_10-15-34_[all]_[5]_[smote]_[crnn(lstm)_acc(0.98)]"
SERVED_MODEL_CLASSIFY_BI_LSTM = "classify_mel-spectrogram_2024-03-18_03-27-38_[all]_[5]_[smote]_[crnn(bi-lstm)_acc(0.98)]"
SERVED_MODEL_CLASSIFY_MFCC = (
    "classify_mfcc_2024-03-14_02-16-35_[all]_[5]_[smote]_[crnn(lstm)_acc(0.97)]"
)
SERVED_MODEL_DETECT_LSTM = "detect_mel-spectrogram_2024-03-15_16-36-20-[all]-[4]-[1-1-0.5]-[crnn(lstm)-acc(0.96)]"
# ------------ 0403 ------------
SERVED_MODEL_CLASSIFY_ENST_3 = "classify_mel-spectrogram_2024-04-03_10-37-02_[enst-idmt]_[3]_[smote]_[conv2d+lstm_f1(0.87)]"
SERVED_MODEL_CLASSIFY_ENST_4 = "classify_mel-spectrogram_2024-04-03_16-12-45_[enst-idmt_enst]_[4]_[crnn(ddm)_f1(0.88)]"
SERVED_MODEL_DETECT_EGMD_4 = "detect_mel-spectrogram_2024-04-03_09-06-58-[e-gmd-clean]_[4]_[lr0.001]_[crnn4_f1(0.58)]"

SERVED_MODEL_ALL = [
    {
        "model_name": SERVED_MODEL_CLASSIFY_LSTM,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_CLASSIFY,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 5,
    },
    {
        "model_name": SERVED_MODEL_CLASSIFY_BI_LSTM,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_CLASSIFY,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 5,
    },
    {
        "model_name": SERVED_MODEL_CLASSIFY_MFCC,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_CLASSIFY,
        "feature_type": MFCC,
        "label_cnt": 5,
    },
    {
        "model_name": SERVED_MODEL_DETECT_LSTM,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_DETECT,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 4,
    },
    {
        "model_name": SERVED_MODEL_CLASSIFY_ENST_3,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_CLASSIFY,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 3,
    },
    {
        "model_name": SERVED_MODEL_CLASSIFY_ENST_4,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_CLASSIFY,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 4,
    },
    {
        "model_name": SERVED_MODEL_DETECT_EGMD_4,
        "is_frozen": True,
        "is_stored": True,
        "method_type": METHOD_DETECT,
        "feature_type": MEL_SPECTROGRAM,
        "label_cnt": 4,
    },
]
