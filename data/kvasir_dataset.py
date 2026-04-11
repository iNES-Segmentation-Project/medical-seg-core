"""
data/kvasir_dataset.py

Kvasir-SEG 데이터셋 로더.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[데이터셋 구조]

  data/kvasir/
  ├── images/   ← 1,000장 RGB JPEG (.jpg)
  └── masks/    ← 1,000장 binary mask JPEG (.jpg), 파일명 동일

[Split 전략]

  전체 1,000장을 seed=42로 고정 후 shuffle → 800/100/100 분할.
  M0, M1 두 실험이 완전히 동일한 split을 사용해야 비교가 유효하다.

  | Split | 범위          | 크기  |
  |-------|--------------|-------|
  | train | all_files[:800]  | 800장 |
  | val   | all_files[800:900] | 100장 |
  | test  | all_files[900:]   | 100장 |

[Mask 처리]

  Kvasir-SEG mask는 RGB JPEG로 저장되어 있다.
  JPEG 압축 artifact로 인해 픽셀값이 정확히 0/255가 아닐 수 있어
  threshold=127을 적용한다.

  grayscale 변환 → threshold → binary int32:
    pixel > 127 → 1 (polyp)
    pixel ≤ 127 → 0 (background)

  PaperlikeTransform이 PIL mode "I" (int32)를 요구하므로
  최종 mask를 mode "I"로 변환한다.

[Transform]

  train : PaperlikeTransform(split="train") — 랜덤 증강 적용
  val   : PaperlikeTransform(split="val")  — resize only
  test  : PaperlikeTransform(split="val")  — resize only (동일)
"""

import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import PaperlikeTransform


class KvasirDataset(Dataset):
    """
    Kvasir-SEG PyTorch Dataset.

    Args:
        root       (str):        데이터 루트 경로. images/, masks/ 폴더를 포함해야 함.
                                 예: "data/kvasir"
        split      (str):        "train" | "val" | "test"
        crop_size  (tuple):      (H, W) 출력 크기. Default: (512, 512)
        split_seed (int):        shuffle seed. M0, M1 동일하게 42로 고정. Default: 42

    Returns (per item):
        image : Tensor (3, H, W)  float32, ImageNet normalized
        mask  : Tensor (H, W)     int64,   values ∈ {0, 1}
                                  0 = background, 1 = polyp
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: tuple = (512, 512),
        split_seed: int = 42,
    ):
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test', got '{split}'"
        )

        self.image_dir = os.path.join(root, "images")
        self.mask_dir  = os.path.join(root, "masks")
        self.split     = split

        # ── Step 1: 파일 목록 로드 (정렬하여 재현성 확보) ────────────────────────
        all_files = sorted(os.listdir(self.image_dir))
        assert len(all_files) == 1000, (
            f"Expected 1000 images in {self.image_dir}, got {len(all_files)}. "
            "Kvasir-SEG 데이터셋이 올바르게 배치되었는지 확인하세요."
        )

        # ── Step 2: seed=42 고정 shuffle ─────────────────────────────────────────
        # random.Random(seed) 인스턴스를 사용해 전역 random 상태에 영향 없음.
        # M0, M1 두 실험이 동일한 split을 보장.
        rng = random.Random(split_seed)
        rng.shuffle(all_files)

        # ── Step 3: 800 / 100 / 100 분할 ────────────────────────────────────────
        if split == "train":
            self.files       = all_files[:800]   # 800장
            transform_split  = "train"           # 랜덤 증강
        elif split == "val":
            self.files       = all_files[800:900] # 100장
            transform_split  = "val"             # resize only
        else:  # test
            self.files       = all_files[900:]   # 100장
            transform_split  = "val"             # resize only (증강 없음)

        # ── Step 4: Transform 설정 ────────────────────────────────────────────────
        # E5와 동일한 PaperlikeTransform 사용.
        # train: RandomResize + Pad + RandomCrop + HFlip + ColorJitter → Normalize
        # val/test: Resize → Normalize
        self.transform = PaperlikeTransform(
            size=crop_size,
            split=transform_split,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        Returns:
            image_t : Tensor (3, H, W)  float32 — ImageNet normalized
            mask_t  : Tensor (H, W)     int64   — 0 (background), 1 (polyp)
        """
        fname = self.files[idx]

        # ── Image 로드 ─────────────────────────────────────────────────────────
        # Kvasir-SEG 이미지: RGB JPEG
        image_path = os.path.join(self.image_dir, fname)
        image = Image.open(image_path).convert("RGB")
        # image: PIL RGB, (H_orig, W_orig)

        # ── Mask 로드 → Binary 변환 ───────────────────────────────────────────
        # Kvasir-SEG mask: RGB JPEG (흰색=polyp, 검정=background)
        # JPEG 압축 artifact → threshold=127 적용
        mask_path = os.path.join(self.mask_dir, fname)
        mask_gray = Image.open(mask_path).convert("L")
        # mask_gray: PIL L (uint8), (H_orig, W_orig), values ∈ [0, 255]

        # threshold → binary int32
        # > 127 → 1 (polyp), ≤ 127 → 0 (background)
        mask_np = np.array(mask_gray, dtype=np.int32)   # (H, W) int32
        mask_np = (mask_np > 127).astype(np.int32)      # (H, W), values ∈ {0, 1}

        # PIL mode "I" (int32) 변환 — PaperlikeTransform 요구사항
        # PaperlikeTransform 내부에서 mode="I" 이미지를 NEAREST resize/crop/flip
        mask = Image.fromarray(mask_np, mode="I")
        # mask: PIL I (int32), (H_orig, W_orig), values ∈ {0, 1}

        # ── Transform 적용 ────────────────────────────────────────────────────
        # (PIL RGB, PIL I) → (Tensor float32, Tensor int64)
        image_t, mask_t = self.transform(image, mask)
        # image_t: (3, 512, 512) float32 — ImageNet mean/std 정규화
        # mask_t:  (512, 512)    int64   — values ∈ {0, 1}

        return image_t, mask_t
