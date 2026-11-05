#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-module SAM3 ablation baseline. No task-specific parameter training."""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise RuntimeError("缺少 pycocotools：pip install -U pycocotools") from e

ANNOT_ROOT_DEFAULT = "/root/autodl-tmp/annotations_23"
ASSET_ROOT_DEFAULT = "/root/autodl-tmp/assets"
CHECKPOINT_PATH_DEFAULT = "/root/autodl-tmp/sam3.pt"
AUTO_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
from sam3.trimapsegment import TrimapSegmenter, TrimapSegmentConfig

RESULT_ROOT_DEFAULT = "/root/autodl-tmp/plot_results/results_matting"
EVAL_CACHE_DIR_DEFAULT = "/root/autodl-tmp/plot_results/results_matting/eval_cache_main"

MODEL_NAME = "SGM-SAM"
DIAGNOSTIC_NAME = "sgm_diagnostics.csv"

def parse_args():
    p = argparse.ArgumentParser("SGM-SAM: SAM3 + structure-guided matting ONLY")
    p.add_argument("--eval_subdir", default="ALL", help='如 "6.28/wan" 或 ALL')
    p.add_argument("--annot_root", default=ANNOT_ROOT_DEFAULT)
    p.add_argument("--asset_root", default=ASSET_ROOT_DEFAULT)
    p.add_argument("--result_root", default=RESULT_ROOT_DEFAULT)
    p.add_argument("--eval_cache_dir", default=EVAL_CACHE_DIR_DEFAULT)
    p.add_argument("--checkpoint_path", default=CHECKPOINT_PATH_DEFAULT)
    p.add_argument("--text_prompt", default="pine seedling leaves, needle leaves")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--confidence_threshold", type=float, default=0.45)
    p.add_argument("--single_class_name", default="object")
    p.add_argument("--only_image", default=None, help="可指定 position1.png 或 6.28/wan/position1.png")
    p.add_argument("--limit_images", type=int, default=0, help="仅测试前 N 张；0 表示全部")
    p.add_argument("--cleanup_output_dir", action="store_true")
    p.add_argument("--no_bbox", action="store_true", help="叠加图不绘制红色实例矩形框")

    # SGM-only parameters (same defaults as TAM-SAM)
    p.add_argument("--disable_trimapsegment", action="store_true", help="仅用于原始 SAM3 对照")
    p.add_argument("--trimap_erode_iters", type=int, default=5)
    p.add_argument("--trimap_dilate_iters", type=int, default=5)
    p.add_argument("--trimap_kernel_size", type=int, default=3)
    p.add_argument("--trimap_alpha_threshold", type=float, default=0.5)
    p.add_argument("--fba_weights", "--trimap_fba_weights", dest="fba_weights",
                   default="/root/sam3-main/sam3/third_party/FBA_Matting/FBA.pth")
    p.add_argument("--fba_repo_dir", "--trimap_fba_repo_dir", dest="fba_repo_dir",
                   default="/root/sam3-main/sam3/third_party/FBA_Matting")
    p.add_argument("--fba_scale", "--trimap_fba_scale", dest="fba_scale", type=float, default=1.0)
    p.add_argument("--disable_fba_compositing", action="store_true",
                   help="关闭 FBA Fusion；必须实际加载更新后的 Deep FBA 模型")
    p.add_argument("--require_deep_fba", action="store_true",
                   help="如果加载 Deep FBA 失败则报错，以确保实验口径一致")
    p.add_argument("--trimap_debug", action="store_true")
    return p.parse_args()

# ===================== 保留原代码辅助函数 =====================
def _natural_key(s: str) -> Tuple:
    parts = []
    for tok in re.split(r"(\d+)", s):
        if tok.isdigit():
            parts.append(int(tok))
        else:
            parts.append(tok.lower())
    return tuple(parts)


def _is_checkpoint_json(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts:
        return True
    if p.name.lower().endswith("-checkpoint.json"):
        return True
    return False


def _draw_polygon_mask(w: int, h: int, points: List[List[float]]) -> np.ndarray:
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) >= 3:
        draw.polygon(xy, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def _draw_rectangle_mask(w: int, h: int, points: List[List[float]]) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((h, w), dtype=np.uint8)
    (x0, y0), (x1, y1) = points[0], points[1]
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return _draw_polygon_mask(w, h, poly)


def _draw_circle_mask(w: int, h: int, points: List[List[float]]) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((h, w), dtype=np.uint8)
    (cx, cy), (px, py) = points[0], points[1]
    r = float(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def labelme_json_to_instance_masks(labelme: Dict[str, Any], w: int, h: int) -> List[np.ndarray]:
    shapes = labelme.get("shapes", None)
    if not isinstance(shapes, list):
        raise ValueError("Unsupported annotation JSON: missing 'shapes' (expect LabelMe).")
    masks: List[np.ndarray] = []
    for s in shapes:
        pts = s.get("points", [])
        if not pts:
            continue
        shape_type = s.get("shape_type", "polygon")
        if shape_type == "polygon":
            m = _draw_polygon_mask(w, h, pts)
        elif shape_type == "rectangle":
            m = _draw_rectangle_mask(w, h, pts)
        elif shape_type == "circle":
            m = _draw_circle_mask(w, h, pts)
        else:
            continue
        if int(m.sum()) == 0:
            continue
        masks.append(m)
    return masks


def _encode_rle(mask01: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask01.astype(np.uint8)))
    if isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _mask_to_bbox(mask01: np.ndarray) -> List[float]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, float(x1 - x0 + 1.0), float(y1 - y0 + 1.0)]


def _masks_to_xyxy_boxes(masks_t: torch.Tensor) -> torch.Tensor:
    """
    输入: [N,1,H,W] 或 [N,H,W]
    输出: [N,4]，格式 [x1,y1,x2,y2]
    """
    if masks_t.ndim == 4 and masks_t.shape[1] == 1:
        masks_hw = masks_t[:, 0]
    elif masks_t.ndim == 3:
        masks_hw = masks_t
    else:
        raise ValueError(f"Unexpected masks shape: {tuple(masks_t.shape)}")

    device = masks_hw.device
    n = masks_hw.shape[0]
    boxes = torch.zeros((n, 4), device=device, dtype=torch.float32)
    for i in range(n):
        m = masks_hw[i] > 0
        if not m.any():
            continue
        ys, xs = torch.where(m)
        boxes[i] = torch.tensor(
            [xs.min(), ys.min(), xs.max(), ys.max()],
            device=device,
            dtype=torch.float32,
        )
    return boxes


def _normalize_model_output(output: Dict[str, Any], image_hw: Tuple[int, int], device: str) -> Dict[str, Any]:
    """
    把 processor 输出统一成后处理模块可直接消费的格式：
      masks: [N,1,H,W]
      scores: [N]
      boxes: [N,4] (xyxy)
      masks_logits: [N,1,H,W]
    """
    if "masks" not in output or output["masks"] is None:
        raise RuntimeError("SAM3 output missing key 'masks'.")

    h, w = image_hw
    dev = torch.device(device)
    state = dict(output)

    masks = state["masks"]
    masks_t = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
    masks_t = masks_t.to(dev)

    if masks_t.ndim == 2:
        masks_t = masks_t.unsqueeze(0).unsqueeze(0)
    elif masks_t.ndim == 3:
        # 可能是 [N,H,W] 或 [1,H,W]
        masks_t = masks_t.unsqueeze(1)
    elif masks_t.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected masks ndim: {masks_t.ndim}")

    if masks_t.shape[0] == 0:
        masks_t = torch.zeros((0, 1, h, w), device=dev, dtype=torch.uint8)
    else:
        masks_t = (masks_t > 0).to(torch.uint8)

    n = int(masks_t.shape[0])
    state["masks"] = masks_t

    scores = state.get("scores", None)
    if scores is None:
        state["scores"] = torch.ones((n,), device=dev, dtype=torch.float32)
    else:
        scores_t = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)
        scores_t = scores_t.to(dev).flatten().float()
        if scores_t.numel() < n:
            pad = torch.ones((n - scores_t.numel(),), device=dev, dtype=torch.float32)
            scores_t = torch.cat([scores_t, pad], dim=0)
        state["scores"] = scores_t[:n]

    masks_logits = state.get("masks_logits", None)
    if masks_logits is None:
        state["masks_logits"] = masks_t.float()
    else:
        ml_t = masks_logits if isinstance(masks_logits, torch.Tensor) else torch.as_tensor(masks_logits)
        ml_t = ml_t.to(dev)
        if ml_t.ndim == 2:
            ml_t = ml_t.unsqueeze(0).unsqueeze(0)
        elif ml_t.ndim == 3:
            ml_t = ml_t.unsqueeze(1)
        elif ml_t.ndim != 4:
            ml_t = masks_t.float()
        if ml_t.shape[0] != n:
            ml_t = masks_t.float()
        state["masks_logits"] = ml_t.float()

    boxes = state.get("boxes", None)
    if boxes is None:
        state["boxes"] = _masks_to_xyxy_boxes(masks_t)
    else:
        boxes_t = boxes if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes)
        boxes_t = boxes_t.to(dev).float()
        if boxes_t.ndim != 2 or boxes_t.shape[-1] != 4 or boxes_t.shape[0] != n:
            boxes_t = _masks_to_xyxy_boxes(masks_t)
        state["boxes"] = boxes_t

    return state


def _fix_output_after_postprocess(state: Dict[str, Any], image_hw: Tuple[int, int], device: str) -> Dict[str, Any]:
    """
    某些后处理模块在空结果时会返回 shape 不完全规整的张量，这里统一修正一次。
    """
    h, w = image_hw
    dev = torch.device(device)
    state = dict(state)

    masks = state.get("masks", None)
    if masks is None:
        state["masks"] = torch.zeros((0, 1, h, w), device=dev, dtype=torch.uint8)
    else:
        masks_t = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
        masks_t = masks_t.to(dev)
        if masks_t.ndim == 2:
            masks_t = masks_t.unsqueeze(0).unsqueeze(0)
        elif masks_t.ndim == 3:
            masks_t = masks_t.unsqueeze(1)
        elif masks_t.ndim != 4:
            masks_t = torch.zeros((0, 1, h, w), device=dev, dtype=torch.uint8)
        state["masks"] = (masks_t > 0).to(torch.uint8)

    n = int(state["masks"].shape[0])

    scores = state.get("scores", None)
    if scores is None:
        state["scores"] = torch.ones((n,), device=dev, dtype=torch.float32)
    else:
        scores_t = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)
        scores_t = scores_t.to(dev).flatten().float()
        if scores_t.numel() != n:
            scores_t = torch.ones((n,), device=dev, dtype=torch.float32)
        state["scores"] = scores_t

    ml = state.get("masks_logits", None)
    if ml is None:
        state["masks_logits"] = state["masks"].float()
    else:
        ml_t = ml if isinstance(ml, torch.Tensor) else torch.as_tensor(ml)
        ml_t = ml_t.to(dev)
        if ml_t.ndim == 2:
            ml_t = ml_t.unsqueeze(0).unsqueeze(0)
        elif ml_t.ndim == 3:
            ml_t = ml_t.unsqueeze(1)
        if ml_t.ndim != 4 or ml_t.shape[0] != n:
            ml_t = state["masks"].float()
        state["masks_logits"] = ml_t.float()

    boxes = state.get("boxes", None)
    if boxes is None:
        state["boxes"] = _masks_to_xyxy_boxes(state["masks"])
    else:
        boxes_t = boxes if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes)
        boxes_t = boxes_t.to(dev).float()
        if boxes_t.ndim != 2 or boxes_t.shape[-1] != 4 or boxes_t.shape[0] != n:
            boxes_t = _masks_to_xyxy_boxes(state["masks"])
        state["boxes"] = boxes_t

    return state


def build_gt_coco(items: List[Dict[str, Any]], cat_name: str) -> Dict[str, Any]:
    coco: Dict[str, Any] = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": cat_name}],
    }
    ann_id = 1
    for image_id, it in enumerate(items, start=1):
        img_path: Path = it["image_path"]
        ann_path: Path = it["ann_path"]
        with Image.open(img_path) as im:
            w, h = im.size
        coco["images"].append(
            {
                "id": int(image_id),
                "file_name": str(it["rel_dir"] / img_path.name).replace("\\", "/"),
                "width": int(w),
                "height": int(h),
            }
        )
        labelme = json.loads(ann_path.read_text(encoding="utf-8"))
        masks = labelme_json_to_instance_masks(labelme, w=w, h=h)
        for m in masks:
            coco["annotations"].append(
                {
                    "id": int(ann_id),
                    "image_id": int(image_id),
                    "category_id": 1,
                    "segmentation": _encode_rle(m),
                    "area": int(m.sum()),
                    "bbox": _mask_to_bbox(m),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco


def _run_coco_eval(gt_json_path: str, pred_json_path: str, iou_type: str) -> COCOeval:
    coco_gt = COCO(gt_json_path)
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def run_coco_eval_segmentation(gt_json_path: str, pred_json_path: str) -> Dict[str, float]:
    preds = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))
    if not isinstance(preds, list) or len(preds) == 0:
        return {"mask_mAP_50_95": 0.0, "mask_AP50": 0.0, "mask_AP75": 0.0}

    coco_eval = _run_coco_eval(gt_json_path, pred_json_path, iou_type="segm")
    return {
        "mask_mAP_50_95": float(coco_eval.stats[0]),
        "mask_AP50": float(coco_eval.stats[1]),
        "mask_AP75": float(coco_eval.stats[2]),
    }


def run_coco_eval_bbox(gt_json_path: str, pred_json_path: str) -> Dict[str, float]:
    preds = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))
    if not isinstance(preds, list) or len(preds) == 0:
        return {"bbox_mAP_50_95": 0.0, "bbox_AP50": 0.0, "bbox_AP75": 0.0}

    coco_eval = _run_coco_eval(gt_json_path, pred_json_path, iou_type="bbox")
    return {
        "bbox_mAP_50_95": float(coco_eval.stats[0]),
        "bbox_AP50": float(coco_eval.stats[1]),
        "bbox_AP75": float(coco_eval.stats[2]),
    }


def _safe_build_sam3(device: str, checkpoint_path: str):
    import inspect

    sig = inspect.signature(build_sam3_image_model)
    kwargs = {}

    if "device" in sig.parameters:
        kwargs["device"] = device

    if "checkpoint_path" in sig.parameters:
        kwargs["checkpoint_path"] = checkpoint_path

    # 指定 SAM3 BPE 词表文件
    if "bpe_path" in sig.parameters:
        kwargs["bpe_path"] = "/root/autodl-tmp/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(**kwargs)

    if hasattr(model, "to"):
        try:
            model.to(torch.device(device))
        except Exception:
            pass

    return model


def _safe_build_processor(model, device: str, confidence_threshold: float):
    import inspect

    sig = inspect.signature(Sam3Processor)
    kwargs = {}
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "confidence_threshold" in sig.parameters:
        kwargs["confidence_threshold"] = confidence_threshold
    return Sam3Processor(model, **kwargs)


# ===================== 找图逻辑 =====================
def find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    if not img_dir.exists():
        return None
    for ext in AUTO_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists() and p.is_file():
            return p
        p2 = img_dir / f"{stem}{ext.upper()}"
        if p2.exists() and p2.is_file():
            return p2
    stem_l = stem.lower()
    for p in img_dir.iterdir():
        if p.is_file() and p.stem.lower() == stem_l and p.suffix.lower() != ".json":
            return p
    return None


def build_items(annot_root: Path, asset_root: Path, eval_subdir: str) -> List[Dict[str, Any]]:
    base = annot_root if eval_subdir.upper() == "ALL" else (annot_root / eval_subdir)
    if not base.exists():
        raise FileNotFoundError(f"Annotation folder not found: {base}")

    ann_files: List[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".json":
            continue
        if _is_checkpoint_json(p):
            continue
        ann_files.append(p)
    ann_files = sorted(ann_files, key=lambda x: _natural_key(str(x)))

    items: List[Dict[str, Any]] = []
    seen_images = set()
    for ann in ann_files:
        rel = ann.relative_to(annot_root)
        rel_dir = rel.parent
        stem = ann.stem

        img_dir = (asset_root / rel_dir).resolve()
        img = find_image_for_stem(img_dir, stem)

        if img is None:
            print(f"[WARN] 跳过：无对应图片 -> {ann}")
            continue

        img_key = str(img.resolve())
        if img_key in seen_images:
            print(f"[WARN] 跳过重复图片 -> {img.name}")
            continue
        seen_images.add(img_key)

        items.append(
            {
                "ann_path": ann,
                "image_path": img,
                "rel_dir": rel_dir,
            }
        )
    return items



def _count_instances_from_state(state: Dict[str, Any]) -> int:
    masks = state.get("masks", None)
    if masks is None:
        return 0
    if isinstance(masks, torch.Tensor):
        return int(masks.shape[0])
    return int(np.asarray(masks).shape[0])


def _save_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save heterogeneous diagnostic dictionaries as a UTF-8-SIG CSV."""
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Saved diagnostics CSV: {path}")


def _save_overlay_and_binary(image: Image.Image, state: Dict[str, Any], overlay_path: Path,
                             show_bbox: bool = True) -> None:
    """Preserve original SGM visualization: blue mask overlay and red instance boxes."""
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    masks_t = state.get("masks")
    masks = []
    if masks_t is not None and masks_t.numel() > 0:
        arr = masks_t.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[:, 0]
        elif arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[1:] != (h, w):
            raise ValueError(f"Mask size {arr.shape} != original image {(h, w)}")
        masks = [m.astype(bool) for m in arr > 0 if m.any()]
    union = np.any(np.stack(masks, axis=0), axis=0) if masks else np.zeros((h, w), dtype=bool)
    overlay = rgb.copy()
    if union.any():
        blue = np.array([0, 0, 255], dtype=np.float32)
        overlay[union] = np.clip(0.55 * overlay[union].astype(np.float32) + 0.45 * blue, 0, 255).astype(np.uint8)
    if show_bbox:
        for mask in masks:
            ys, xs = np.where(mask)
            if xs.size:
                cv2.rectangle(overlay, (int(xs.min()), int(ys.min())),
                              (int(xs.max()), int(ys.max())), (255, 0, 0), 2)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(
        overlay_path.with_name(overlay_path.stem + "_binary.png"))

def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU")
        device = "cpu"

    annot_root = Path(args.annot_root)
    asset_root = Path(args.asset_root)
    result_root = Path(args.result_root)
    cache_root = Path(args.eval_cache_dir) / args.eval_subdir.replace("/", "_")
    cache_root.mkdir(parents=True, exist_ok=True)

    items = build_items(annot_root, asset_root, args.eval_subdir)
    if args.only_image:
        selected = args.only_image.replace("\\", "/").lstrip("./")
        items = [it for it in items if it["image_path"].name == selected
                 or str(it["rel_dir"] / it["image_path"].name).replace("\\", "/") == selected
                 or (not Path(selected).suffix and it["image_path"].stem == selected)]
    if args.limit_images < 0:
        raise ValueError("--limit_images must be non-negative")
    if args.limit_images:
        items = items[:args.limit_images]
    if not items:
        raise RuntimeError("没有找到对应的已标注图像，请检查 annot_root、asset_root、eval_subdir 和 only_image")

    print(f"[INFO] {MODEL_NAME}: annotated images={len(items)}")
    if args.cleanup_output_dir:
        output_dir = result_root if args.eval_subdir.upper() == "ALL" else result_root / args.eval_subdir
        if output_dir.exists():
            for file in output_dir.rglob("*"):
                if file.is_file():
                    file.unlink()
            print(f"[WARN] Cleaned image outputs under {output_dir}")

    gt = build_gt_coco(items, cat_name=args.single_class_name)
    gt_json_path = cache_root / "gt_coco.json"
    gt_json_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    model = _safe_build_sam3(device=device, checkpoint_path=args.checkpoint_path)
    processor = _safe_build_processor(model, device=device, confidence_threshold=args.confidence_threshold)
    segmenter = None
    if not args.disable_trimapsegment:
        cfg = TrimapSegmentConfig(
            erode_iters=args.trimap_erode_iters,
            dilate_iters=args.trimap_dilate_iters,
            kernel_size=args.trimap_kernel_size,
            alpha_threshold=args.trimap_alpha_threshold,
            fba_weights=args.fba_weights,
            fba_repo_dir=args.fba_repo_dir,
            fba_scale=args.fba_scale,
            use_fba_fusion=not args.disable_fba_compositing,
        )
        segmenter = TrimapSegmenter(config=cfg, device=device)
        matting_mode = ("deep_fba" if segmenter._use_deep_fba else
                        "pymatting" if segmenter._pymatting_ok else "original_mask_fallback")
        print(f"[INFO] SGM loaded: method={matting_mode}, FBA Fusion={cfg.use_fba_fusion}")
        if (args.disable_fba_compositing or args.require_deep_fba) and not segmenter._use_deep_fba:
            raise RuntimeError(
                "需要真正加载修改后的 Deep FBA 才能进行 Fusion ON/OFF 消融。"
                "请检查 FBA.pth、repo_dir、networks/models.py 的 use_fba_fusion 接口。"
            )
    else:
        print("[INFO] SGM disabled; running SAM3 control")
    pred_results: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    for image_id, item in enumerate(items, 1):
        img_path = item["image_path"]
        rel_dir = item["rel_dir"]
        rel_name = str(rel_dir / img_path.name).replace("\\", "/")
        out_path = result_root / rel_dir / img_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.text_prompt)
        output = _normalize_model_output(output, image_hw=(h, w), device=device)
        raw_n = _count_instances_from_state(output)
        if segmenter is not None:
            if args.trimap_debug:
                print(f"[DEBUG] {rel_name}: raw masks={raw_n}")
            output = segmenter.process(image, output)
            diagnostics_rows.extend({"file_name": rel_name, **row} for row in segmenter.last_diagnostics)
            output = _fix_output_after_postprocess(output, image_hw=(h, w), device=device)
        after_n = _count_instances_from_state(output)
        if args.trimap_debug:
            print(f"[DEBUG] {rel_name}: after SGM={after_n}")
        _save_overlay_and_binary(image, output, out_path, show_bbox=not args.no_bbox)
        masks_t = output["masks"]
        scores_t = output["scores"]
        if masks_t.ndim == 4:
            masks_t = masks_t[:, 0]
        saved = 0
        for idx in range(masks_t.shape[0]):
            m = (masks_t[idx].detach().cpu().numpy() > 0).astype(np.uint8)
            if not m.any():
                continue
            score = float(scores_t[idx].detach().cpu().item())
            pred_results.append({
                "image_id": int(image_id), "category_id": 1,
                "segmentation": _encode_rle(m), "bbox": _mask_to_bbox(m),
                "score": score,
            })
            saved += 1
        per_image_rows.append({"file_name": rel_name, "image_id": image_id,
                               "raw_instances": raw_n, "final_instances": saved})
        print(f"[OK] {rel_name}: raw={raw_n}, after_module={after_n}, saved={saved}")

    pred_json_path = cache_root / "pred_coco.json"
    pred_json_path.write_text(json.dumps(pred_results, ensure_ascii=False), encoding="utf-8")
    _save_rows_csv(diagnostics_rows, cache_root / DIAGNOSTIC_NAME)
    _save_rows_csv(per_image_rows, cache_root / "per_image_counts.csv")
    metrics = {
        **run_coco_eval_segmentation(str(gt_json_path), str(pred_json_path)),
        **run_coco_eval_bbox(str(gt_json_path), str(pred_json_path)),
    }
    metrics["model_variant"] = MODEL_NAME
    metrics["evaluated_images"] = len(items)
    metrics["predicted_instances"] = len(pred_results)
    metrics["experiment_config"] = vars(args)
    (cache_root / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== {MODEL_NAME} / COCO =====")
    for key in ["mask_mAP_50_95", "mask_AP50", "mask_AP75",
                "bbox_mAP_50_95", "bbox_AP50", "bbox_AP75"]:
        print(f"{key:20s}: {metrics[key]:.4f}")
    print(f"[OK] Evaluation: {cache_root}")
    print("[NEXT] Boundary F1: python boundary_eval.py --gt_json .../gt_coco.json --pred_json .../pred_coco.json")


if __name__ == "__main__":
    main()
