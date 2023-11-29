import os

"""
-- save checkpoint path
"""
# Include the epoch in the file name (uses `str.format`)
# checkpoint_path = "drum_cnn_model_ckpt/drum_cnn-{epoch:04d}.ckpt"
checkpoint_path = "drum_cnn_model_ckpt/drum_cnn.ckpt"


"""
-- 상수
"""
# -- 일정한 시간 간격으로 음압을 측정하는 주파수, 44100Hz
SAMPLE_RATE = 44100

# -- onset duration
ONSET_DURATION = 0.1

# -- dir name
PATTERN = 'pattern'
PER_DRUM = 'per_drum'

# -- delay
DELAY = 30000

# -- 결과가 0.5 이상이면 1, 아니면 0
predict_standard = 0.5

# -- batch
batch_size = 20

"""
-- drum mapping

-- 파일 이름 형식
-- per_drum : CC_04_9949.wav
-- pattern : P1_08_0001.wav
"""
code2drum = {0:'CC', 1:'HH', 2:'RC', 3:'ST', 4:'MT', 5:'SD', 6:'FT', 7:'KK'}
# -- {'CC':0, 'HH':1, ...}
drum2code = {v: k for k, v in code2drum.items()}
# -- {'CC':[1,0,0,0,0,0,0,0], 'HH':[0,1,0,0,0,0,0,0], ...}
onehot_drum2code = {}
for code, index in drum2code.items():
    drum_mapping = [0] * len(drum2code)
    drum_mapping[index] = 1
    onehot_drum2code[code] = drum_mapping

pattern = {'HH':onehot_drum2code['HH'],
           'SD':onehot_drum2code['SD'],
           'HH_KK':[0,1,0,0,0,0,0,1],
           'HH_SD':[0,1,0,0,0,1,0,0]}

p_hh_kk = pattern['HH_KK']
p_sd = pattern['SD']
p_hh = pattern['HH']
p_hh_sd = pattern['HH_SD']

p1_2code = [p_hh_kk, p_hh, p_hh_sd, p_hh, p_hh_kk, p_hh_kk, p_hh_sd, p_hh]
p2_2code = [p_hh_kk, p_hh, p_hh, p_hh, p_sd, p_hh, p_hh, p_hh, p_hh_kk, p_hh, p_hh, p_hh, p_sd, p_hh, p_hh, p_hh]
pattern2code = {'P1':p1_2code, 'P2':p2_2code}
