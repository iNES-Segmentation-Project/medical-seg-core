## Project Overview

SegFormer-B0 기반 General Purpose Segmentation 파이프라인의 의료 도메인 적용 실험 저장소.

CamVid(도시 도메인)에서 설계 및 검증된 E5 구조(FPN + CE+Dice+Boundary)가
의료 도메인(Kvasir-SEG, 대장내시경)에서도 기본 구조 대비 우위를 보이는지 검증한다.

- Encoder: SegFormer-B0 구조 고정 (수정 불가)
- Decoder: MLP (baseline) vs FPN
- Loss: CE / CE+Dice+Boundary
- Dataset: Kvasir-SEG (2 classes) — Train 800 / Val 100 / Test 100

---

## 상위 프로젝트

| 저장소 | 설명 |
|--------|------|
| [segmentation-core](https://github.com/iNES-Segmentation-Project/segformer-core) | CamVid 기반 E0~E5 실험 (완결) |
| [segmentation-docs](https://github.com/iNES-Segmentation-Project/segformer-docs) | 전체 실험 설계 문서 및 결과 보고서 |
| **medical-seg-core** | Kvasir-SEG 기반 M0, M1 실험 (현재 저장소) |

---

## Project Structure

```
medical-seg-core/
├── models/
│   ├── encoder/              # MiT-B0 (segmentation-core와 동일)
│   ├── decoder/              # mlp.py, fpn.py
│   └── loss/                 # ce, dice, boundary
├── data/
│   ├── transforms.py         # PaperlikeTransform (segmentation-core와 동일)
│   ├── kvasir_dataset.py
│   └── kvasir/
│       ├── images/
│       └── masks/
├── configs/
│   ├── m0_mlp_ce.yaml
│   └── m1_fpn_cedice_boundary.yaml
├── train.py
├── evaluate.py
└── utils/
```

---

## Setup

```bash
# 1. 데이터셋 준비
mkdir -p data/kvasir/images data/kvasir/masks
# kvasir-seg.zip 다운로드 후 images/, masks/ 각 경로로 이동
# https://datasets.simula.no/downloads/kvasir-seg.zip

# 2. 환경 설치
pip install torch torchvision transformers timm pyyaml
```

---

## Training & Evaluation

```bash
python train.py --config configs/m0_mlp_ce.yaml
python train.py --config configs/m1_fpn_cedice_boundary.yaml

python evaluate.py --config configs/m1_fpn_cedice_boundary.yaml \
                   --checkpoint weights/m1_best.pth
```

---

## Experiments

| ID | Encoder | Decoder | Loss | 파라미터 수 | 역할 |
|----|---------|---------|------|------------|------|
| M0 | MiT-B0 | MLP | CE | 3.7M | 대조군 (기본 구조) |
| M1 | MiT-B0 | FPN | CE+Dice+Boundary | 6.1M | 핵심 실험 (E5 구조) |

변수는 구조 하나다. 출발 가중치, 증강, 학습 전략은 완전히 동일하게 고정했다.

고정 설정: PaperlikeTransform / warmup_poly / AdamW / enc lr 6e-5, dec lr 6e-4 / max 100 epoch / early stopping patience 20

---

## Results

### Val / Test 성능 비교

| ID | Best Val Dice | Best Val Epoch | Test Dice | Test mIoU | Val→Test 변화 |
|----|---------------|----------------|-----------|-----------|--------------|
| M0 | 0.8998 | 70 | 0.9222 | 0.9147 | +0.0224 |
| M1 | **0.9083** | **91** | **0.9275** | **0.9201** | +0.0192 |
| M1 - M0 | +0.0085 | — | +0.0053 | +0.0054 | — |

Val→Test 변화가 양수(+)인 것은 test 성능이 val보다 높게 나온 것이다.
test 100장이 상대적으로 쉬운 샘플로 구성되었거나,
val 기준 best 저장 전략이 test에서 일반화된 것으로 해석한다.

### Per-class IoU (Test 기준)

| 클래스 | M0 | M1 | M1 - M0 |
|--------|----|----|---------|
| background | 0.9737 | 0.9755 | +0.0018 |
| polyp | 0.8557 | **0.8647** | +0.0090 |

### 수렴 비교

| 구분 | M0 | M1 |
|------|----|----|
| Val Dice 0.80 최초 달성 epoch | 5 | 4 |
| Val Dice 0.90 최초 달성 epoch | — | 52 |
| Best 달성 epoch | 70 | 91 |
| Early stopping | epoch 90 발동 | 미발동 (100 epoch 완주) |

### CamVid E5 vs Kvasir M1

| 항목 | CamVid E5 | Kvasir M1 |
|------|-----------|-----------|
| 도메인 | 도시 도로 (11 class) | 대장내시경 (2 class) |
| 구조 | FPN + CE+Dice+Boundary | 동일 |
| 학습 전략 | warmup_poly, diff LR, PaperlikeTransform | 동일 |
| Best Val mIoU/Dice | 0.8043 | 0.9083 |
| Test mIoU/Dice | 0.7572 | 0.9275 |

---

## Key Findings

**E5 구조는 의료 도메인에서도 기본 구조 대비 일관된 우위를 보인다.**

Test Dice M1(0.9275) > M0(0.9222), polyp IoU M1(0.8647) > M0(0.8557).
차이는 크지 않으나 방향성이 일치하며, CamVid에서의 결과(FPN > MLP)와 동일한 패턴이다.

**도메인 전환 성공 기준(Test Dice ≥ 0.80)을 M0, M1 모두 크게 상회했다.**

M0 0.9222, M1 0.9275로 기준치 대비 0.12 이상 초과.
ImageNet pretrain encoder의 전이 학습 효과가 의료 도메인에서도 강하게 작동함을 보여준다.

**E5 파이프라인은 도메인에 관계없이 재사용 가능하다.**

CamVid에서 설계된 구조, 증강, 학습 전략을 그대로 Kvasir-SEG에 적용하여 합리적인 성능을 달성했다.
config 하나만 교체하면 새 도메인에서 바로 실험 가능한 범용성이 검증되었다.

---

## Notes

- Encoder는 SegFormer-B0와 기능적으로 동일하게 유지한다.
- M0, M1은 구조(Decoder + Loss) 외 모든 설정을 동일하게 고정한다.
- M1은 E5 구조의 범용성 검증 실험이며, E5와 직접 성능 비교 대상이 아니다.
- MMSegmentation 의존성 없이 순수 PyTorch로 구현한다.