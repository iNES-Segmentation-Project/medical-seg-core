# medical-seg-core

SegFormer-B0 기반 General Purpose Segmentation 파이프라인의 의료 도메인 적용 실험 저장소.

CamVid(도시 도메인)에서 설계 및 검증된 E5 구조(FPN + CE+Dice+Boundary)가  
의료 도메인(Kvasir-SEG, 대장내시경)에서도 기본 구조 대비 우위를 보이는지 검증한다.

---

## 상위 프로젝트

| 저장소 | 설명 |
|--------|------|
| [segmentation-core](https://github.com/iNES-Segmentation-Project/segformer-core) | CamVid 기반 E0~E5 실험 (완결) |
| [segmentation-docs](https://github.com/iNES-Segmentation-Project/segformer-docs) | 전체 실험 설계 문서 및 결과 보고서 |
| **medical-seg-core** | Kvasir-SEG 기반 M0, M1 실험 (현재 저장소) |

---

## 실험 목적

E5 구조가 **도메인에 관계없이 범용적으로 우수한 구조**임을 입증한다.

```
E5: FPN + CE+Dice+Boundary + ImageNet pretrain → CamVid 학습  (기존)
M1: FPN + CE+Dice+Boundary + ImageNet pretrain → Kvasir 학습  (이번)
```

구조, 증강, 학습 전략은 E5와 완전히 동일하다. 차이는 학습 도메인(데이터셋)뿐이다.

---

## 실험 구성

| 실험 | Decoder | Loss | 출발 가중치 | 역할 |
|------|---------|------|------------|------|
| M0 | MLP | CE | ImageNet pretrain | 대조군 (기본 구조) |
| M1 | FPN | CE + Dice + Boundary | ImageNet pretrain | 핵심 실험 (E5 구조) |

---

## 데이터셋

**Kvasir-SEG** — RGB 대장내시경 polyp segmentation 데이터셋

| 항목 | 내용 |
|------|------|
| 전체 이미지 수 | 1,000장 |
| 클래스 | 2 (polyp / background) |
| Split | train 800 / val 100 / test 100 (seed=42) |
| Mask 형식 | pixel-level binary mask |
| 라이선스 | 연구/교육 목적 무료 |
| 다운로드 | https://datasets.simula.no/downloads/kvasir-seg.zip |

---

## 프로젝트 구조

```
medical-seg-core/
├── configs/
│   ├── dataset/
│   │   └── kvasir.yaml
│   ├── m0_mlp_ce.yaml
│   └── m1_fpn_cedice_boundary.yaml
├── models/
│   ├── encoder/              # MiT-B0 (segmentation-core와 동일)
│   ├── decoder/              # mlp.py, fpn.py
│   └── loss/                 # ce, dice, boundary
├── data/
│   ├── transforms.py         # PaperlikeTransform (segmentation-core와 동일)
│   ├── kvasir_dataset.py     # Kvasir-SEG DataLoader
│   └── kvasir/
│       ├── images/           # 1,000장 (직접 다운로드)
│       └── masks/            # 1,000장
├── train.py
├── evaluate.py
└── utils/
```

---

## 고정 설정

M0, M1 공통으로 아래 설정을 동일하게 유지한다.

| 항목 | 값 |
|------|-----|
| Augmentation | PaperlikeTransform (E5와 동일) |
| Scheduler | warmup_poly |
| Optimizer | AdamW |
| Learning Rate | encoder 6e-5 / decoder 6e-4 |
| Max Epoch | 100 |
| Early Stopping | patience 20 (val Dice 기준) |
| Crop size | 512×512 |
| 출발 가중치 | ImageNet pretrain (nvidia/mit-b0) |

---

## 빠른 시작

### 1. 데이터셋 준비

```bash
# kvasir-seg.zip 다운로드 후 압축 해제
mkdir -p data/kvasir/images data/kvasir/masks
# images/, masks/ 폴더 내용을 각각 위 경로로 이동
```

### 2. 환경 설치

```bash
pip install torch torchvision transformers timm pyyaml
```

### 3. 학습 실행

```bash
# M0 (대조군)
python train.py --config configs/m0_mlp_ce.yaml

# M1 (E5 구조)
python train.py --config configs/m1_fpn_cedice_boundary.yaml
```

### 4. 평가

```bash
python evaluate.py --config configs/m1_fpn_cedice_boundary.yaml \
                   --checkpoint weights/m1_best.pth
```

---

## 성공 기준

| 판정 | 기준 |
|------|------|
| 도메인 전환 성공 | M1 Test Dice ≥ 0.80 |
| 구조 범용성 확인 | M1 Test Dice > M0 Test Dice |

---

## 참고 문헌

- Xie et al., SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, NeurIPS 2021
- Jha et al., Kvasir-SEG: A Segmented Polyp Dataset, MMM 2020
- Lin et al., Feature Pyramid Networks for Object Detection, CVPR 2017