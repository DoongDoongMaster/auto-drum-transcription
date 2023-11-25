# rename-dataset

드럼 녹음 데이터셋 파일 이름 바꾸는 코드

# HOW TO RUN

1. Creation of virtual environments
   ```bash
   python -m venv .venv
   ```
1. Pip install package
   ```bash
   pip install -r requirements.txt
   ```
1. Run python code
   ```bash
   python main.py
   ```

# 드럼 데이터셋 네임 규칙

```
{DRUM}_{BEAT}_0001.m4a
```

# DRUM

- `CC`: 크래쉬 심벌
- `HH`: 하이헷
- `RC`: 라이드 심벌
- `ST`: 스몰탐
- `MT`: 미들탐
- `SD`: 스네어
- `FT`: 플로어탐
- `KK`: 킥
- `P1`: 패턴 1
- `P2`: 패턴 2

[참고](https://iewha-my.sharepoint.com/:x:/r/personal/jiyoung_06_i_ewha_ac_kr/_layouts/15/Doc.aspx?sourcedoc=%7B0455C17B-0AE5-4A32-AF51-305230A91591%7D&file=Labeling.xlsx&action=default&mobileredirect=true)

# BEAT

- `04`: 4분음표
- `08`: 8분음표
- `16`: 16분음표
