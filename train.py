"""
train.py

Kvasir-SEG M0 / M1 실험 학습 스크립트.

실행:
  python train.py --config configs/m0_mlp_ce.yaml
  python train.py --config configs/m1_fpn_cedice_boundary.yaml

출력:
  weights/{experiment_name}/best_model.pth   ← best val Dice 기준
  logs/{experiment_name}/train_log.csv        ← epoch별 학습 기록

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[학습 파이프라인]

  1. Config 로드 (YAML)
  2. 모델 빌드 → ImageNet pretrained encoder 로드
  3. KvasirDataset (800 train / 100 val)
  4. Optimizer: AdamW with differential LR
       encoder: 6e-5  /  decoder: 6e-4
  5. Scheduler: warmup_poly (per epoch)
       - warmup_epochs=10: linear 0 → base_lr
       - poly decay: (1 - progress)^0.9
  6. 학습 루프:
       - train_loss (batch avg)
       - val Dice / val mIoU (전체 epoch 집계)
       - best model 저장 (val Dice 기준)
       - early stopping (patience=20)
  7. CSV 로그 저장

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Metric 계산 방법]

  Confusion Matrix를 배치마다 누적 후 epoch 끝에 집계.
  - Dice : 2*TP[polyp] / (2*TP[polyp] + FP[polyp] + FN[polyp])
  - mIoU : mean(TP_c / (TP_c + FP_c + FN_c))  for c in {0, 1}
"""

import os
import csv
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segformer import build_segformer_b0, build_segformer_b0_fpn
from models.loss.cross_entropy import CrossEntropyLoss
from models.loss.combined_loss import CombinedLoss
from data.kvasir_dataset import KvasirDataset
from utils.checkpoint import load_pretrained_encoder


# =============================================================================
# ── 모델 / Loss 빌더 ──────────────────────────────────────────────────────────
# =============================================================================

def build_model(cfg: dict) -> torch.nn.Module:
    """
    Config 기반 SegFormer 모델 빌드.

    decoder 키:
      "mlp" → build_segformer_b0  (M0)
      "fpn" → build_segformer_b0_fpn (M1)
    """
    decoder_type = cfg["model"]["decoder"]
    num_classes  = cfg["dataset"]["num_classes"]
    dropout      = cfg["model"].get("dropout", 0.1)

    if decoder_type == "mlp":
        embed_dim = cfg["model"].get("embed_dim", 256)
        model = build_segformer_b0(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    elif decoder_type == "fpn":
        fpn_dim = cfg["model"].get("fpn_dim", 256)
        model = build_segformer_b0_fpn(
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown decoder type: '{decoder_type}'. Choose 'mlp' or 'fpn'.")

    return model


def build_criterion(cfg: dict) -> torch.nn.Module:
    """
    Config 기반 Loss 함수 빌드.

    loss.mode 키:
      "ce"               → CrossEntropyLoss                (M0)
      "ce+dice+boundary" → CombinedLoss(mode=...)          (M1)
    """
    loss_cfg     = cfg["loss"]
    mode         = loss_cfg["mode"]
    num_classes  = cfg["dataset"]["num_classes"]
    ignore_index = loss_cfg.get("ignore_index", 255)

    if mode == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)

    elif mode in ("ce+dice", "ce+boundary", "ce+dice+boundary"):
        return CombinedLoss(
            mode=mode,
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            aux_weight=loss_cfg.get("aux_weight", 1.0),
            boundary_weight=loss_cfg.get("boundary_weight", 0.1),
        )

    else:
        raise ValueError(
            f"Unknown loss mode: '{mode}'. "
            "Choose 'ce', 'ce+dice', 'ce+boundary', or 'ce+dice+boundary'."
        )


# =============================================================================
# ── LR Scheduler ──────────────────────────────────────────────────────────────
# =============================================================================

def make_warmup_poly_lambda(warmup_epochs: int, max_epochs: int, poly_power: float):
    """
    Warmup + Poly decay LR lambda 함수 생성.

    epoch < warmup_epochs : linear warmup
      lr_factor = (epoch + 1) / warmup_epochs   [1/warmup → 1.0]

    epoch >= warmup_epochs : poly decay
      progress  = (epoch - warmup) / (max_epochs - warmup)
      lr_factor = (1 - progress)^poly_power

    LambdaLR에서 lr = base_lr * lr_lambda(epoch) 로 적용됨.

    Args:
        warmup_epochs : linear warmup 구간 epoch 수
        max_epochs    : 전체 최대 epoch
        poly_power    : poly decay 지수

    Returns:
        lr_lambda (callable): epoch → float
    """
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Linear warmup: 0 → base_lr
            # epoch=0 → 1/warmup_epochs, epoch=warmup_epochs-1 → 1.0
            return (epoch + 1) / warmup_epochs
        else:
            # Poly decay
            remaining = max(max_epochs - warmup_epochs, 1)
            progress  = (epoch - warmup_epochs) / remaining
            return max(0.0, (1.0 - progress) ** poly_power)

    return lr_lambda


# =============================================================================
# ── Metric 계산 ───────────────────────────────────────────────────────────────
# =============================================================================

@torch.no_grad()
def accumulate_conf_matrix(
    conf: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    배치별 confusion matrix를 누적한다. epoch 끝에 Dice/mIoU 계산에 사용.

    Args:
        conf     : (num_classes, num_classes) int64 — 누적 중인 confusion matrix
        logits   : (B, C, H, W) float32 — raw logits
        targets  : (B, H, W)    int64   — class index
        num_classes  : int
        ignore_index : int

    Returns:
        conf : 업데이트된 (num_classes, num_classes) confusion matrix
    """
    # (B, C, H, W) → argmax → (B, H, W)
    pred = logits.argmax(dim=1)  # (B, H, W)

    # Flatten: (B*H*W,)
    pred    = pred.reshape(-1)
    targets = targets.reshape(-1)

    # ignore_index 픽셀 제거
    valid   = (targets != ignore_index)
    pred    = pred[valid]
    targets = targets[valid]

    # confusion matrix 누적:
    #   conf[true_class, pred_class] += 1
    # scatter_add_를 사용해 반복문 없이 O(N) 처리
    idx = num_classes * targets + pred                         # 1D linear index
    conf.reshape(-1).scatter_add_(
        0, idx, torch.ones_like(idx)
    )

    return conf


def metrics_from_conf(conf: torch.Tensor):
    """
    Confusion matrix에서 Dice / mIoU / per-class IoU 계산.

    Kvasir-SEG binary segmentation:
      - class 0: background
      - class 1: polyp  ← Dice score의 기준 클래스

    Args:
        conf : (num_classes, num_classes) int64
               conf[true, pred]

    Returns:
        dice      : float — Dice score for class 1 (polyp)
        miou      : float — mean IoU across all classes
        iou_list  : list[float] — per-class IoU [background, polyp]
    """
    conf = conf.float()

    tp = conf.diag()                       # (C,) — 각 클래스 TP
    fp = conf.sum(dim=0) - conf.diag()    # (C,) — 예측은 c인데 GT는 아닌 경우
    fn = conf.sum(dim=1) - conf.diag()    # (C,) — GT는 c인데 예측이 틀린 경우

    # Per-class IoU: TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn + 1e-6)      # (C,)
    miou = iou.mean().item()

    # Dice for class 1 (polyp): 2*TP / (2*TP + FP + FN)
    dice = (2.0 * tp[1] / (2.0 * tp[1] + fp[1] + fn[1] + 1e-6)).item()

    return dice, miou, iou.tolist()


# =============================================================================
# ── 학습 / 검증 루프 ──────────────────────────────────────────────────────────
# =============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    1 epoch 학습.

    Returns:
        avg_loss : float — epoch 평균 train loss
    """
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)   # (B, 3, H, W) float32
        masks  = masks.to(device)    # (B, H, W)    int64

        optimizer.zero_grad()

        logits = model(images)       # (B, num_classes, H, W) float32
        loss   = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = 255,
):
    """
    전체 val set 평가. Confusion matrix를 배치마다 누적 후 집계.

    Returns:
        val_dice : float — Dice score (polyp class)
        val_miou : float — mean IoU
        iou_list : list[float] — per-class IoU
    """
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)   # (B, 3, H, W)
        masks  = masks.to(device)    # (B, H, W)

        logits = model(images)       # (B, num_classes, H, W)
        conf   = accumulate_conf_matrix(conf, logits, masks, num_classes, ignore_index)

    dice, miou, iou_list = metrics_from_conf(conf)
    return dice, miou, iou_list


# =============================================================================
# ── Main ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train M0 / M1 on Kvasir-SEG")
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML config 경로. 예: configs/m0_mlp_ce.yaml"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=None,
        help="epoch 수 override (dry-run 등 테스트용). 미지정 시 config 값 사용."
    )
    args = parser.parse_args()

    # ── Config 로드 ───────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --max_epoch 지정 시 config 값을 override
    if args.max_epoch is not None:
        cfg["train"]["max_epochs"] = args.max_epoch

    print(f"\n{'='*60}")
    print(f"[Train] Experiment : {cfg['experiment']['name']}")
    print(f"[Train] Config     : {args.config}")
    print(f"{'='*60}")

    # ── Device 설정 ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device     : {device}")
    if device.type == "cuda":
        print(f"[Train] GPU        : {torch.cuda.get_device_name(0)}")

    # ── Config 값 추출 ────────────────────────────────────────────────────────
    dataset_cfg  = cfg["dataset"]
    train_cfg    = cfg["train"]

    num_classes  = dataset_cfg["num_classes"]
    ignore_index = dataset_cfg.get("ignore_index", 255)
    crop_size    = tuple(dataset_cfg["crop_size"])    # (512, 512)
    split_seed   = dataset_cfg.get("split_seed", 42)

    max_epochs    = train_cfg["max_epochs"]
    batch_size    = train_cfg["batch_size"]
    num_workers   = train_cfg.get("num_workers", 4)
    encoder_lr    = train_cfg["encoder_lr"]
    decoder_lr    = train_cfg["decoder_lr"]
    weight_decay  = train_cfg.get("weight_decay", 0.01)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    poly_power    = train_cfg.get("poly_power", 0.9)
    patience      = train_cfg.get("early_stop_patience", 20)
    save_dir      = train_cfg["save_dir"]

    # ── 모델 빌드 → Pretrained Encoder 로드 ──────────────────────────────────
    print("\n[Train] Building model...")
    model = build_model(cfg)
    model = load_pretrained_encoder(model)   # nvidia/mit-b0 ImageNet 가중치
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] Total params: {total_params:.1f}M")

    # ── Loss 함수 ─────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg)

    # ── DataLoader ────────────────────────────────────────────────────────────
    print("\n[Train] Loading datasets...")
    train_ds = KvasirDataset(
        root=dataset_cfg["root"],
        split="train",
        crop_size=crop_size,
        split_seed=split_seed,
    )
    val_ds = KvasirDataset(
        root=dataset_cfg["root"],
        split="val",
        crop_size=crop_size,
        split_seed=split_seed,
    )
    print(f"[Train] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,     # 마지막 불완전 배치 제거 (BN 안정성)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer (Differential LR) ───────────────────────────────────────────
    # encoder: 6e-5 (pretrained 가중치 보존)
    # decoder: 6e-4 (새로 학습, 10× 큰 lr)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )

    # ── LR Scheduler: Warmup + Poly ───────────────────────────────────────────
    # LambdaLR: lr = base_lr * lr_lambda(epoch)
    # 두 param group에 동일한 lambda 적용 (각자의 base_lr 기준으로 곱해짐)
    lr_lambda = make_warmup_poly_lambda(warmup_epochs, max_epochs, poly_power)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda, lr_lambda],  # [encoder group, decoder group]
    )

    # ── 저장 경로 설정 ────────────────────────────────────────────────────────
    exp_name = cfg["experiment"]["name"]
    log_dir  = os.path.join("logs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    log_path   = os.path.join(log_dir, "train_log.csv")
    log_fields = ["epoch", "train_loss", "val_dice", "val_miou",
                  "iou_bg", "iou_polyp", "encoder_lr", "decoder_lr"]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    print(f"\n[Train] Save dir : {save_dir}")
    print(f"[Train] Log path : {log_path}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    best_val_dice    = 0.0
    patience_counter = 0

    print(f"\n[Train] Start training ({max_epochs} epochs, patience={patience})\n")

    for epoch in range(1, max_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ── Validate ──────────────────────────────────────────────────────────
        val_dice, val_miou, val_ious = validate(
            model, val_loader, num_classes, device, ignore_index
        )

        # ── LR Step (per epoch) ───────────────────────────────────────────────
        scheduler.step()
        enc_lr = optimizer.param_groups[0]["lr"]
        dec_lr = optimizer.param_groups[1]["lr"]

        # ── 출력 ──────────────────────────────────────────────────────────────
        print(
            f"[Epoch {epoch:03d}/{max_epochs}] "
            f"loss={train_loss:.4f}  "
            f"val_dice={val_dice:.4f}  val_miou={val_miou:.4f}  "
            f"iou=[{val_ious[0]:.3f},{val_ious[1]:.3f}]  "
            f"enc_lr={enc_lr:.2e}  dec_lr={dec_lr:.2e}"
        )

        # ── CSV 로깅 ──────────────────────────────────────────────────────────
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_dice":   f"{val_dice:.6f}",
                "val_miou":   f"{val_miou:.6f}",
                "iou_bg":     f"{val_ious[0]:.6f}",
                "iou_polyp":  f"{val_ious[1]:.6f}",
                "encoder_lr": f"{enc_lr:.2e}",
                "decoder_lr": f"{dec_lr:.2e}",
            })

        # ── Best model 저장 + Early Stopping ─────────────────────────────────
        if val_dice > best_val_dice:
            best_val_dice    = val_dice
            patience_counter = 0

            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "cfg":           cfg,
            }, ckpt_path)
            print(f"  → [Best] val_dice={best_val_dice:.4f}  saved: {ckpt_path}")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n[Train] Early stopping: no improvement for {patience} epochs.\n"
                    f"[Train] Best val_dice = {best_val_dice:.4f} (epoch saved)"
                )
                break

    # ── 완료 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Train] Done.  Experiment : {exp_name}")
    print(f"[Train] Best val Dice     : {best_val_dice:.4f}")
    print(f"[Train] Checkpoint        : {os.path.join(save_dir, 'best_model.pth')}")
    print(f"[Train] Log               : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
