# 딥러닝 스터디

##  구성(Top-level)
- 본 저장소는 **Jupyter Notebook (`.ipynb`)** 파일들로 구성되어 있습니다.
- 일부 노트북은 공개 데이터셋(예: **이안류 CCTV**, **차량 파손**)을 전제로 하며, 데이터 파일은 저장소에 포함되지 않을 수 있습니다.

##  노트북 목차
아래 표에서 각 노트북을 확인할 수 있습니다. (로컬/깃 저장소에서는 파일명을 클릭하여 열 수 있습니다.)

| Notebook | Path |
|---|---|
| 1. 컴퓨터 비전.ipynb | `./1. 컴퓨터 비전.ipynb` |
| 2. Object Detection.ipynb | `./2. Object Detection.ipynb` |
| 4. 이안류 CCTV 데이터셋.ipynb | `./4. 이안류 CCTV 데이터셋.ipynb` |
| 5. Segmentation.ipynb | `./5. Segmentation.ipynb` |
| 5. Segmentation+.ipynb | `./5. Segmentation+.ipynb` |
| 6. 차량 파손 데이터셋.ipynb | `./6. 차량 파손 데이터셋.ipynb` |
| 7.비지도 학습.ipynb | `./7.비지도 학습.ipynb` |
| 8. 오토인코더.ipynb | `./8. 오토인코더.ipynb` |
| 9. GAN.ipynb | `./9. GAN.ipynb` |
| 10. 자연어 처리.ipynb | `./10. 자연어 처리.ipynb` |
| 11. 벡터화.ipynb | `./11. 벡터화.ipynb` |
| 12. 신경망 기반의 벡터화.ipynb | `./12. 신경망 기반의 벡터화.ipynb` |
| 13. RNN.ipynb | `./13. RNN.ipynb` |
| 14. LSRM과 GRU.ipynb | `./14. LSRM과 GRU.ipynb` |
| 15. Seq2Seq.ipynb | `./15. Seq2Seq.ipynb` |
| 16. 어텐션 메커니즘.ipynb | `./16. 어텐션 메커니즘.ipynb` |

##  환경 준비
권장: **conda** 또는 **venv** 사용

```bash
# (선택) conda 환경
conda create -n dl-notes python=3.10 -y
conda activate dl-notes

# Jupyter 및 필수 라이브러리
pip install --upgrade pip
pip install jupyterlab notebook
pip install numpy pandas matplotlib seaborn opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1 환경일 때
# CPU-only 또는 다른 CUDA 버전은 https://pytorch.org 에서 설치 커맨드를 확인하세요.
pip install ultralytics  # YOLOv8
pip install nltk gensim
```

>  **NLTK** 사용 노트북은 최초 실행 시 다음과 같이 리소스를 내려받아야 할 수 있습니다:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## ▶ 실행 방법
1. 저장소를 클론하거나 ZIP을 풀어 로컬 디렉터리를 준비합니다.
2. (위) 환경을 설치하고 활성화합니다.
3. Jupyter Lab/Notebook 실행:
   ```bash
   jupyter lab
   # 또는
   jupyter notebook
   ```
4. 브라우저에서 각 노트북(`.ipynb`)을 열고 **위에서 아래로** 순차 실행하세요.

##  데이터셋 안내
- `4. 이안류 CCTV 데이터셋.ipynb` / `6. 차량 파손 데이터셋.ipynb` / Segmentation 관련 노트북은 **이미지/동영상 데이터** 경로를 요구할 수 있습니다.
- 데이터 경로 예시:
  - `data/ianryu_cctv/*` (예시)
  - `data/car_damage/*` (예시)
- 본 저장소에는 데이터가 포함되지 않을 수 있으니, 노트북 상단의 **데이터 로드 셀**을 확인하고 경로를 수정하세요.

##  주요 토픽별 개요
- **컴퓨터 비전 기초:** 이미지 표현, 변환, 시각화.
- **Object Detection (YOLOv8/Ultralytics):** 데이터셋 구성, 학습/추론, 평가 지표.
- **Segmentation:** 마스크 생성/평가, 실습(+ 확장 버전).
- **비지도 학습:** 표본 시각화/클러스터링 등(노트북: *비지도 학습*).
- **오토인코더 & GAN:** 생성 모델의 학습과 샘플링.
- **NLP(벡터화 → RNN/GRU → Seq2Seq → 어텐션):** 토큰화/임베딩, 순환신경망 계열, 시퀀스 변환, 주의 메커니즘.

##  의존성(요약)
다음 라이브러리 사용을 확인했습니다.
- python>=3.10
- jupyterlab
- notebook
- numpy
- pandas
- matplotlib
- seaborn
- opencv-python
- torch
- torchvision
- ultralytics
- nltk
- gensim

> ⚠️ 실제 사용 버전은 환경에 따라 달라질 수 있습니다. 재현성 확보가 필요하다면 `requirements.txt` 또는 `environment.yml`를 고정 버전으로 생성하시길 권장합니다.

##  재현 팁
- **랜덤 시드 고정**: `numpy`, `torch`의 시드를 고정해 실험 재현성을 높일 수 있습니다.
- **GPU 가속**: CUDA 가용 시 `torch.cuda.is_available()`를 확인하고, 텐서를 GPU로 이동하세요.
- **체크포인트**: 학습이 오래 걸리는 실습은 모델 가중치를 주기적으로 저장하세요.
