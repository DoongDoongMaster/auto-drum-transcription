# Automatic Drum Transcription (ADT)
>드럼의 박자를 인식하고 악기를 분류하는 드럼 전사 모델

## 📝 Table of Contents
- [Introduction](https://github.com/DoongDoongMaster/automatic-drum-transcription/edit/main/README.md#introduction)
- [Dataset](https://github.com/DoongDoongMaster/automatic-drum-transcription/edit/main/README.md#dataset)
- [Preparation](https://github.com/DoongDoongMaster/automatic-drum-transcription/edit/main/README.md#preparation)
- [Run](https://github.com/DoongDoongMaster/automatic-drum-transcription/edit/main/README.md#run)
<br>

## Introduction
### 🥁ADT 란?
드럼 오디오 신호에서 음악적 기호를 추출해, 악보 형태로 재구성하는 것

### 👩‍🔬연구 방법
<img src = "https://github.com/DoongDoongMaster/automatic-drum-transcription/assets/68186101/5218b993-55e3-41a5-b525-4130c36165e7" width="100%" height="100%">

#### Segment and Classify
: 드럼 오디오 신호 강도를 기반으로 Onset 을 인식하여 오디오 조각으로 자르고, 각 조각의 특성에서 패턴을 인식

#### Separate and Detect
: 오디오 신호를 드럼 악기의 특성에 따라 분류하고, 각 악기에 대한 활성화 함수로부터 Peak-Picking 을 거쳐 Onset 을 추출

### ❓Why
드럼 전사 모델을 통해 사용자의 드럼 연주를 인식해서, 이를 기반으로 정답 악보와 비교해 채점을 하는 기능을 위해 필요한 기술
<br><br>

## Dataset
- ADT 연구 분야에서 보편적으로 사용되는 데이터셋 사용
- 어쿠스틱 드럼셋에 맞춰 clean 과정을 거침
- 온셋을 기준으로 동시에 친 음을 판단 후, 다음 온셋이 나오기 전까지 음을 잘라서 토큰으로 만드는 과정을 거침

|데이터셋|길이|토큰 개수|
|------|---|---|
|[MDB](https://github.com/CarlSouthall/MDBDrums)|35 m|3,638 개|
|[IDMT](https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html)|116 m|10,489 개|
|[E-GMD](https://magenta.tensorflow.org/datasets/e-gmd)|26,374 m|126,608 개|
|[ENST](https://perso.telecom-paristech.fr/grichard/ENST-drums/)|194 m|23,503 개|

<br><br>

## Preparation
1. 데이터셋 아래 폴더 경로로 다운로드
  ```
  automatic-drum-transcription
  ├── data
  │   ├── raw
  │   │   ├── MDBDrums
  │   │   ├── IDMT-SMT-DRUMS-V2
  │   │   ├── e-gmd-v1.0.0
  │   │   ├── ENST-drums-public
  ```
2. create conda env
  ```shell
  conda env create -f drum.yml
  ```

<br><br>

## Run
### 1. Feature Extraction
- 오디오 조각으로 자른 후, `Mel-Spectrogram` 으로 피쳐 추출
- 실행 코드 (`src/` 위치에서 실행)
  ```python
  data_paths = DataProcessing.get_paths({데이터 경로})
  FeatureExtractor.feature_extractor(data_paths, METHOD_DETECT, MEL_SPECTROGRAM, PKL)
  FeatureExtractor.load_feature_file(METHOD_DETECT, MEL_SPECTROGRAM, PKL, ENST, TRAIN)
  ```
### 2. Training
- `Segment and Classify`, `Separate and Detect` 중 원하는 모델 클래스 선택해서 학습 진행 가능 
- 실행 코드 (`src/` 위치에서 실행)
  ```python
  # == split_data, label_type 매개변수 바꿔서 사용!
  split_data = {
      TRAIN: [MDB, IDMT, ENST, E_GMD],
      VALIDATION: [MDB, IDMT, ENST, E_GMD],
      TEST: [MDB, IDMT, ENST, E_GMD],
  }

  # 모델 객체 생성
  segment_classify = SegmentClassifyModel(
      training_epochs=50,
      batch_size=8,
      opt_learning_rate=0.001,
      feature_type=MEL_SPECTROGRAM,
  )
  # split dataset
  segment_classify.create_dataset(split_data, group_dict=CLASSIFY_TYPES)
  # create model
  segment_classify.create()
  # train model
  segment_classify.train()
  ```
### 3. Evaluate
- 실행 코드 (`src/` 위치에서 실행)
  ```python
  segment_classify.evaluate() # 모델 평가
  segment_classify.save() # 모델 저장
  ```
### 4. Inferenct
- 실행 코드 (`src/` 위치에서 실행)
  ```python
  segment_classify = SegmentClassifyModel(feature_type=MEL_SPECTROGRAM)
  segment_classify.predict({예측할 wav 파일 경로}, {bpm 정보}, 0)
  ```

