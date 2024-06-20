# Automatic Drum Transcription (ADT)
>드럼의 박자를 인식하고 악기를 분류하는 드럼 전사 모델

## 📝 Table of Contents
- [Introduction]()
- [Dataset]()
- [Experiment]()
- []()
- []()


## 1. Background
### 🥁 ADT 란?
드럼 오디오 신호에서 음악적 기호를 추출해, 악보 형태로 재구성하는 것

### 👩‍🔬 연구 방법
<img src = "https://github.com/DoongDoongMaster/automatic-drum-transcription/assets/68186101/5218b993-55e3-41a5-b525-4130c36165e7" width="100%" height="100%">

#### Segment and Classify
: 드럼 오디오 신호 강도를 기반으로 Onset 을 인식하여 오디오 조각으로 자르고, 각 조각의 특성에서 패턴을 인식

#### Separate and Detect
: 오디오 신호를 드럼 악기의 특성에 따라 분류하고, 각 악기에 대한 활성화 함수로부터 Peak-Picking 을 거쳐 Onset 을 추출



