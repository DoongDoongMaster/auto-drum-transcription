"""
-- data path
"""
ROOT_PATH = "../data"
RAW_PATH = "raw"
NEW_PATH = "new"
PROCESSED_FEATURE = "processed-feature"
METHOD_CLASSIFY = "classify"
METHOD_DETECT = "detect"
METHOD_RHYTHM = "rhythm"

"""
-- data origin
"""
DDM_OWN = "ddm-own"
IDMT = "IDMT-SMT-DRUMS-V2"
ENST = "ENST-drums-public"
E_GMD = "e-gmd-v1.0.0"


"""
-- save checkpoint path
"""
CHECKPOINT_PATH = "../models"


"""
-- related audio
"""
# -- 일정한 시간 간격으로 음압을 측정하는 주파수, 44100Hz
SAMPLE_RATE = 44100

# -- onset duration
ONSET_DURATION = 0.02


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
FEATURE_PARAM = {
    METHOD_CLASSIFY: {
        MFCC: {
            "n_features": 40,
            "n_times": 20,
            "n_channels": 1,
            "n_classes": len(CODE2DRUM),
            "hop_length": 512,
        },
        STFT: {
            "n_times": 1024,
            "n_fft": 2048,
            "n_classes": len(CODE2DRUM),
            "hop_length": 512,
            "win_length": 2048,
        },
        MEL_SPECTROGRAM: {
            "n_times": 1024,
            "n_fft": 2048,
            "n_classes": len(CODE2DRUM),
            "hop_length": 512,
            "win_length": 2048,
        },
    },
    METHOD_DETECT: {
        MFCC: {
            "n_features": 40,
            "n_times": 512,
            "n_channels": 1,
            "n_classes": len(CODE2DRUM),
            "hop_length": 512,
        },
        STFT: {
            "n_times": 512,
            "n_fft": 1024,
            "n_classes": len(CODE2DRUM),
            "hop_length": 128,
            "win_length": 512,
        },
        MEL_SPECTROGRAM: {
            "n_times": 512,
            "n_fft": 1024,
            "n_classes": len(CODE2DRUM),
            "hop_length": 128,
            "win_length": 512,
        },
    },
    METHOD_RHYTHM: {
        MFCC: {
            "n_features": 40,
            "n_times": 512,
            "n_channels": 1,
            "n_classes": len(CODE2DRUM),
            "hop_length": 512,
        },
        STFT: {
            "n_times": 1024,
            "n_fft": 2048,
            "n_classes": 1,
            "hop_length": 512,
            "win_length": 1024,
        },
        # -- adt 논문 참고 파라미터
        MEL_SPECTROGRAM: {
            "n_times": 1024,
            "n_fft": 2048,  # -- FFT window length
            "n_classes": len(CODE2DRUM),
            "hop_length": 441,
            "win_length": 1024,
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
MILLISECOND = 1000000
