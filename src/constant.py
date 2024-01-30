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
DDM_OWN = "ddm-own"
IDMT = "IDMT-SMT-DRUMS-V2"
ENST = "ENST-drums-public"
E_GMD = "e-gmd-v1.0.0"


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

# -- 오디오 자를 시, onset 기준 양 옆으로 몇 초씩 자를지 (단위: sec)
ONSET_DURATION = 0.1

# -- onset offset: int (onset position 양 옆으로 몇 개씩 붙일지)
ONSET_OFFSET = 0

# -- chunk time
CHUNK_LENGTH = 12

# -- unit (1 sec)
MILLISECOND = 1000000

# -- classify method duration
CLASSIFY_DURATION = 0.2


"""
-- drum mapping

-- 파일 이름 형식
-- per_drum : CC_04_9949.wav
-- pattern : P1_08_0001.wav
"""
CODE2DRUM = {0: "CC", 1: "HH", 2: "RC", 3: "ST", 4: "MT", 5: "SD", 6: "FT", 7: "KK"}
# -- {'CC':0, 'HH':1, ...}
DRUM2CODE = {v: k for k, v in CODE2DRUM.items()}
# -- {'CC':[1,0,0,0,0,0,0,0], 'HH':[0,1,0,0,0,0,0,0], ...}
ONEHOT_DRUM2CODE = {}
for code, index in DRUM2CODE.items():
    drum_mapping = [0] * len(DRUM2CODE)
    drum_mapping[index] = 1
    ONEHOT_DRUM2CODE[code] = drum_mapping

PATTERN = {
    "HH": ONEHOT_DRUM2CODE["HH"],
    "SD": ONEHOT_DRUM2CODE["SD"],
    "HH_KK": [0, 1, 0, 0, 0, 0, 0, 1],
    "HH_SD": [0, 1, 0, 0, 0, 1, 0, 0],
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
