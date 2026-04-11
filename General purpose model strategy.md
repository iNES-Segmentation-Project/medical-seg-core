# General Purpose Model 확보 전략 (최종 v3)

---

## 1. General Purpose Model의 정의

이 프로젝트에서 General Purpose Model은 특정 가중치가 아니라 **구조와 파이프라인**을 의미한다.

```
General Purpose = E5 구조 (FPN + CE+Dice+Boundary)
                + PaperlikeTransform 기반 학습 파이프라인
                + config 교체만으로 새 데이터셋에 적용 가능한 인터페이스
```

누군가 새로운 데이터셋을 가져왔을 때, config만 수정하고 ImageNet pretrain 가중치에서 출발해 fine-tuning하면 기본 구조보다 합리적으로 더 나은 결과가 나오는 구조 자체가 랩실 자산이다.

가중치는 도메인마다 별도로 학습한다. CamVid E5 가중치는 그 구조가 실제로 작동한다는 첫 번째 증거이며, 이번 Kvasir-SEG 실험이 두 번째 증거가 된다.

---

## 2. 실험 목적

E5 구조(FPN + CE+Dice+Boundary)가 CamVid(도시 도메인)에서만 유효한 것인지, 의료 도메인에서도 기본 구조 대비 우위를 보이는지 검증한다.

이를 통해 E5 구조가 **도메인에 관계없이 범용적으로 우수한 구조**임을 입증하는 것이 목표다.

---

## 3. 실험 설계

### 핵심 구조

| 실험 | Decoder | Loss | 출발 가중치 | 역할 |
|------|---------|------|------------|------|
| M0 | MLP | CE | ImageNet pretrain | 대조군 (기본 구조) |
| M1 | FPN | CE+Dice+Boundary | ImageNet pretrain | 핵심 실험 (E5 구조) |

변수는 **구조 하나**다. 출발 가중치, 증강, 학습 전략은 M0와 M1 모두 동일하게 고정한다.

### E5와 M1의 관계

```
E5: FPN + CE+Dice+Boundary + ImageNet pretrain → CamVid 학습
M1: FPN + CE+Dice+Boundary + ImageNet pretrain → Kvasir 학습
```

E5와 M1은 구조, 증강, 학습 전략이 완전히 동일하다. 차이는 학습 도메인(데이터셋)뿐이다.
이것이 "파이프라인은 그대로인데 도메인이 바뀌어도 작동한다"는 주장의 핵심 근거가 된다.

---

## 4. 데이터셋 확정 — Kvasir-SEG

| 항목 | 내용 |
|------|------|
| 모달리티 | RGB 대장내시경 (colonoscopy) |
| 전체 이미지 수 | 1,000장 |
| 클래스 수 | 2 (polyp / background) |
| Mask 형식 | pixel-level binary mask |
| 라이선스 | 연구/교육 목적 무료, 상업 사용 금지 |
| 다운로드 | https://datasets.simula.no/downloads/kvasir-seg.zip |

### 데이터 Split

공식 데이터셋은 이미지와 마스크 폴더만 제공하며, train/val/test 폴더가 별도로 구분되어 있지 않다. 전체 1,000장에서 직접 분할해야 한다.

| Split | 이미지 수 | 비율 |
|-------|----------|------|
| Train | 800장 | 80% |
| Val | 100장 | 10% |
| Test | 100장 | 10% |

M0와 M1이 완전히 동일한 split을 사용해야 비교가 유효하다. seed=42로 고정한다.

```python
import random
random.seed(42)
all_files = sorted(os.listdir(image_dir))
random.shuffle(all_files)
train_files = all_files[:800]
val_files   = all_files[800:900]
test_files  = all_files[900:]
```

---

## 5. 고정 설정

M0, M1 두 실험 모두 아래 설정을 동일하게 유지한다.

| 항목 | 값 |
|------|-----|
| 데이터셋 | Kvasir-SEG (train 800 / val 100 / test 100) |
| num_classes | 2 |
| 출발 가중치 | ImageNet pretrain |
| Augmentation | PaperlikeTransform (E5와 동일) |
| Scheduler | warmup_poly (E5와 동일) |
| Optimizer | AdamW |
| Learning Rate | encoder 6e-5 / decoder 6e-4 (differential LR) |
| Max Epoch | 100 |
| Best model 기준 | val Dice score 최고점 |
| Early stopping | patience 20 epoch |
| Crop size | 512×512 |
| 평가 지표 | Dice score, mIoU (train/val/test 모두 기록) |

---

## 6. 검증 결과 기록 구조

M0, M1 각각에 대해 train/val/test 세 단계 결과를 모두 기록한다.

| 단계 | 기록 항목 |
|------|-----------|
| Train | epoch별 loss, val Dice score 추이 |
| Val | best val Dice score (best model 저장 기준) |
| Test | test Dice score, test mIoU (best model로 평가) |

최종 비교표 형태는 다음과 같다.

| 실험 | Best Val Dice | Test Dice | Test mIoU | Val→Test 하락폭 |
|------|--------------|-----------|-----------|----------------|
| M0 (MLP + CE) | | | | |
| M1 (FPN + CE+Dice+Boundary) | | | | |

---

## 7. 성공 기준

| 판정 | 기준 |
|------|------|
| 도메인 전환 성공 | M1 Test Dice score 0.80 이상 |
| 구조 범용성 확인 | M1 Test Dice > M0 Test Dice |

두 조건이 모두 충족되면 "E5 구조는 도시 도메인뿐 아니라 의료 도메인에서도 기본 구조 대비 우수하다. 따라서 랩실 범용 구조로 채택할 근거가 있다"는 주장이 성립한다.

Kvasir-SEG 기준 현재 SOTA는 Dice 0.90~0.92 수준이나, 경량 모델(B0) 및 fine-tuning 조건에서 0.80 이상이면 도메인 전환 가능성의 근거로 충분하다.

---

## 8. 결과 해석 방향

**M1 > M0이고 M1 Test Dice 0.80 이상인 경우**

E5 구조(FPN + 복합 Loss)가 의료 도메인에서도 기본 구조(MLP + CE) 대비 우위를 보였다. CamVid에서 검증된 구조적 우수성이 도메인이 달라져도 재현되었으며, E5 파이프라인을 랩실 범용 구조로 채택할 근거가 확보된다.

**M1 > M0이나 M1 Test Dice 0.80 미만인 경우**

구조적 우위는 확인되었으나 절대 성능이 기준에 미치지 못했다. epoch 수 확대 또는 더 큰 encoder(B2 이상) 도입을 후속 과제로 제안한다.

**M1 <= M0인 경우**

이진 분류 태스크에서 복합 Loss의 이점이 상쇄된 것으로 해석 가능하다. 단순 태스크에서는 CE 단독이 오히려 안정적일 수 있으며, multi-class와 binary 태스크 간 Loss 전략 차이로 논의할 수 있다.

---

## 9. GitHub 저장소 구조

```
segmentation-lab/ (organization)
├── segformer-core     ← CamVid E0~E5 (완결, 보존, 수정 금지)
├── segformer-docs       ← 전체 허브 문서
└── medical-seg-core        ← Kvasir-SEG M0, M1 실험 (신규)
```

---

## 10. 로컬 세팅 — `C:\segmentation-project\medical-seg-core`

### 10-1. 새 repo에서 새로 만들 것

| 파일 | 설명 |
|------|------|
| `configs/dataset/kvasir.yaml` | num_classes, 경로, mean/std, split seed 등 |
| `configs/m0_mlp_ce.yaml` | M0 실험 전용 config |
| `configs/m1_fpn_cedice_boundary.yaml` | M1 실험 전용 config |
| `data/kvasir_dataset.py` | DataLoader + 800/100/100 split 로직 |
| `README.md` | repo 설명, 실험 재현 방법 |

**kvasir.yaml 작성 예시**
```yaml
dataset:
  name: kvasir
  root: data/kvasir
  num_classes: 2
  ignore_index: 255
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  crop_size: [512, 512]
  split_seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

**kvasir_dataset.py 핵심 구현 포인트**
```python
# 1. 전체 1000장 파일 리스트 로드
# 2. seed=42로 shuffle 후 800/100/100 분할
# 3. split 인자(train/val/test)에 따라 해당 subset 반환
# 4. mask binary 변환: 픽셀값 > 127 → 1, 나머지 → 0
# 5. E5와 동일한 PaperlikeTransform 적용
```

### 10-2. segformer-core에서 그대로 가져올 것

수정 없이 그대로 복사해서 사용 가능한 파일들이다.

| 복사 대상 | 용도 |
|-----------|------|
| `models/encoder/` 전체 | MiT-B0 encoder (수정 없음) |
| `models/decoder/mlp.py` | M0용 decoder |
| `models/decoder/fpn.py` | M1용 decoder |
| `models/loss/ce_loss.py` | M0, M1 공통 |
| `models/loss/dice_loss.py` | M1용 |
| `models/loss/boundary_loss.py` | M1용 (호환 확인 후) |
| `data/transforms.py` | PaperlikeTransform 그대로 |
| `utils/` 전체 | 유틸리티 그대로 |

### 10-3. 가져오되 수정이 필요한 것

| 파일 | 수정 내용 | 이유 |
|------|-----------|------|
| `train.py` | dataset 분기 추가 (kvasir 분기), num_classes=2 반영 | CamVid 전용 코드 제거 필요 |
| `evaluate.py` | **Dice score 계산 로직 추가** | binary segmentation 평가 필수. 현재 mIoU만 있을 가능성 높음 |
| `models/loss/boundary_loss.py` | binary mask 입력(2클래스) 호환 여부 확인 및 수정 | E5는 11클래스 기준으로 구현되어 있을 수 있음 |
| `models/loss/ce_loss.py` | ignore_index=255 처리 binary 호환 확인 | 동작은 하지만 binary 입력 시 검증 필요 |

**evaluate.py Dice score 추가 예시**
```python
def dice_score(pred, target, smooth=1e-6):
    # pred: (B, 2, H, W) logits → argmax → binary
    pred = pred.argmax(dim=1).float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
```

### 10-4. segmentation-core에서 가져오지 않을 것

| 파일/폴더 | 이유 |
|-----------|------|
| `configs/e0~e5_*.yaml` | CamVid 전용, 불필요 |
| `data/camvid_dataset.py` | CamVid 전용, 불필요 |
| `weights/` | M0, M1 모두 ImageNet pretrain에서 새로 시작 |

### 10-5. 세팅 명령어 순서

```bash
# 1. 새 repo 클론
cd C:\segmentation-project
git clone https://github.com/segmentation-lab/medical-seg-core
cd medical-seg-core

# 2. segmentation-core에서 필요한 파일 복사
xcopy ..\segformer-core\models models\ /E /I
xcopy ..\segformer-core\data\transforms.py data\
xcopy ..\segformer-core\utils utils\ /E /I
copy ..\ssegformer-core\train.py .
copy ..\ssegformer-cor\evaluate.py .

# 3. Kvasir-SEG 다운로드 후 폴더 구조 정리
mkdir data\kvasir\images
mkdir data\kvasir\masks
# kvasir-seg.zip 압축 해제 후
# images/ → data\kvasir\images\
# masks/  → data\kvasir\masks\
# (images/와 masks/ 파일명이 동일해야 함)
```

---

## 11. 진행 순서

| 순서 | 작업 |
|------|------|
| 1 | medical-seg-core repo 생성 및 로컬 클론 |
| 2 | segformer-core에서 파일 복사 |
| 3 | Kvasir-SEG 다운로드 및 폴더 정리 |
| 4 | kvasir_dataset.py 작성 (split 로직 포함) |
| 5 | evaluate.py에 Dice score 추가 |
| 6 | boundary_loss.py binary 호환 확인 및 수정 |
| 7 | config 파일 작성 (kvasir.yaml, m0, m1) |
| 8 | M0 실험 (MLP + CE) — train/val/test 전체 기록 |
| 9 | M1 실험 (FPN + CE+Dice+Boundary) — train/val/test 전체 기록 |
| 10 | 결과 정리 및 segmentation-docs 업데이트 |