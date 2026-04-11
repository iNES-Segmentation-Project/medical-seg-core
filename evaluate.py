"""
evaluate.py

Kvasir-SEG M0 / M1 실험 평가 스크립트.

best_model.pth를 로드하여 val 또는 test set 전체를 평가한다.

실행:
  # test set 평가 (기본)
  python evaluate.py --config configs/m0_mlp_ce.yaml

  # val set 평가
  python evaluate.py --config configs/m0_mlp_ce.yaml --split val

  # checkpoint 직접 지정
  python evaluate.py --config configs/m1_fpn_cedice_boundary.yaml \\
                     --checkpoint weights/m1_fpn_cedice_boundary/best_model.pth

출력 예시:
  ==================================================
  [Results] split=test  config=configs/m1_fpn_cedice_boundary.yaml
  ==================================================
    Dice score (polyp) : 0.8312
    mIoU               : 0.7654
    Per-class IoU:
      class 0 (background ): 0.9201
      class 1 (polyp      ): 0.6107
  ==================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Metric 정의]

  Dice score (polyp):
    2 * TP[1] / (2 * TP[1] + FP[1] + FN[1])
    - 클래스 1 (polyp)에 대한 F1 score
    - 이진 분류 세그멘테이션 핵심 지표

  mIoU:
    mean( TP_c / (TP_c + FP_c + FN_c) )  for c in {0, 1}
    - 모든 클래스 평균 IoU

  Per-class IoU:
    class 0 (background): TP[0] / (TP[0] + FP[0] + FN[0])
    class 1 (polyp)     : TP[1] / (TP[1] + FP[1] + FN[1])

  모두 Confusion Matrix를 배치마다 누적 후 test set 전체 기준으로 집계.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.kvasir_dataset import KvasirDataset

# train.py에서 공통 유틸 재사용
from train import (
    build_model,
    accumulate_conf_matrix,
    metrics_from_conf,
)


# =============================================================================
# ── 평가 ─────────────────────────────────────────────────────────────────────
# =============================================================================

@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = 255,
):
    """
    DataLoader 전체를 평가하여 Dice / mIoU / per-class IoU를 반환.

    Args:
        model        : 평가할 SegFormer 모델 (eval mode)
        loader       : val 또는 test DataLoader
        num_classes  : 클래스 수 (Kvasir-SEG = 2)
        device       : torch.device
        ignore_index : 무시할 class index (default: 255)

    Returns:
        dice     : float — Dice score (polyp class=1)
        miou     : float — mean IoU
        iou_list : list[float] — per-class IoU [background, polyp]
    """
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for images, masks in tqdm(loader, desc="  evaluate", leave=False):
        images = images.to(device)   # (B, 3, H, W) float32
        masks  = masks.to(device)    # (B, H, W)    int64

        logits = model(images)       # (B, num_classes, H, W) float32

        # Confusion matrix 배치 누적
        conf = accumulate_conf_matrix(
            conf, logits, masks, num_classes, ignore_index
        )

    dice, miou, iou_list = metrics_from_conf(conf)
    return dice, miou, iou_list


# =============================================================================
# ── Main ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate M0 / M1 on Kvasir-SEG")
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML config 경로. 예: configs/m0_mlp_ce.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="checkpoint 경로. 미지정 시 save_dir/best_model.pth 사용"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="평가할 split. 기본값: test"
    )
    args = parser.parse_args()

    # ── Config 로드 ───────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]

    print(f"\n{'='*60}")
    print(f"[Evaluate] Experiment : {exp_name}")
    print(f"[Evaluate] Config     : {args.config}")
    print(f"[Evaluate] Split      : {args.split}")
    print(f"{'='*60}")

    # ── Device 설정 ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device     : {device}")

    # ── Config 값 추출 ────────────────────────────────────────────────────────
    dataset_cfg  = cfg["dataset"]
    train_cfg    = cfg["train"]

    num_classes  = dataset_cfg["num_classes"]
    ignore_index = dataset_cfg.get("ignore_index", 255)
    crop_size    = tuple(dataset_cfg["crop_size"])
    split_seed   = dataset_cfg.get("split_seed", 42)
    batch_size   = train_cfg["batch_size"]
    num_workers  = train_cfg.get("num_workers", 4)

    # ── Checkpoint 경로 결정 ──────────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(
        train_cfg["save_dir"], "best_model.pth"
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "먼저 train.py로 학습을 완료하거나 --checkpoint 경로를 지정하세요."
        )

    # ── 모델 빌드 → Checkpoint 로드 ──────────────────────────────────────────
    print(f"\n[Evaluate] Loading model...")
    model = build_model(cfg)
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    saved_epoch    = ckpt.get("epoch", "?")
    saved_val_dice = ckpt.get("best_val_dice", float("nan"))
    print(f"[Evaluate] Checkpoint : {ckpt_path}")
    print(f"[Evaluate] Saved epoch: {saved_epoch}")
    print(f"[Evaluate] Best val Dice (from ckpt): {saved_val_dice:.4f}")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    ds = KvasirDataset(
        root=dataset_cfg["root"],
        split=args.split,
        crop_size=crop_size,
        split_seed=split_seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Evaluate] {args.split} size: {len(ds)}")

    # ── 평가 실행 ─────────────────────────────────────────────────────────────
    print(f"\n[Evaluate] Running evaluation on {args.split} set...")
    dice, miou, iou_list = evaluate_split(
        model, loader, num_classes, device, ignore_index
    )

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    class_names = ["background", "polyp"]

    print(f"\n{'='*50}")
    print(f"[Results] split={args.split}  config={args.config}")
    print(f"{'='*50}")
    print(f"  Dice score (polyp) : {dice:.4f}")
    print(f"  mIoU               : {miou:.4f}")
    print(f"  Per-class IoU:")
    for i, (name, iou_val) in enumerate(zip(class_names, iou_list)):
        print(f"    class {i} ({name:<12s}): {iou_val:.4f}")
    print(f"{'='*50}\n")

    # ── 성공 기준 판정 (test split일 때만) ────────────────────────────────────
    if args.split == "test":
        print("[Judgment]")
        if dice >= 0.80:
            print(f"  ✓ 도메인 전환 성공: Test Dice {dice:.4f} ≥ 0.80")
        else:
            print(f"  ✗ 도메인 전환 기준 미달: Test Dice {dice:.4f} < 0.80")
        print(
            "  (M1 vs M0 비교는 두 config 각각의 결과를 확인하세요)\n"
        )


if __name__ == "__main__":
    main()
