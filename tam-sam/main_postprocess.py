import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.p_visualization_utils import plot_results

# ====== 后处理模块导入 ======
from sam3.postprocessor import SeedlingPostProcessor

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise RuntimeError("缺少 pycocotools：pip install -U pycocotools") from e

ANNOT_ROOT_DEFAULT = "/root/autodl-tmp/annotations_89"
ASSET_ROOT_DEFAULT = "/root/autodl-tmp/assets"
RESULT_ROOT_DEFAULT = "/root/autodl-tmp/plot_results/results_postprocess"
EVAL_CACHE_DIR_DEFAULT = "/root/autodl-tmp/plot_results/results_postprocess/eval_cache_main"
CHECKPOINT_PATH_DEFAULT = "/root/autodl-tmp/sam3.pt"
AUTO_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def parse_args():
    p = argparse.ArgumentParser(
        "SAM3 batch inference + postprocessor + COCO mask/bbox AP eval"
    )
    p.add_argument("--eval_subdir", default="ALL", help='如 "6.4/zao" 或 ALL')
    p.add_argument("--annot_root", default=ANNOT_ROOT_DEFAULT)
    p.add_argument("--asset_root", default=ASSET_ROOT_DEFAULT)
    p.add_argument("--result_root", default=RESULT_ROOT_DEFAULT)
    p.add_argument("--eval_cache_dir", default=EVAL_CACHE_DIR_DEFAULT)
    p.add_argument("--checkpoint_path", default=CHECKPOINT_PATH_DEFAULT)
    p.add_argument("--text_prompt", default="pine seedling leaves, needle leaves")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--confidence_threshold", type=float, default=0.45)
    p.add_argument("--single_class_name", default="object")
    p.add_argument(
        "--cleanup_output_dir",
        action="store_true",
        help="若启用：先清空 result_root/eval_subdir 对应的输出文件夹（谨慎）",
    )

    # ====== postprocessor 参数 ======
    p.add_argument("--disable_postprocessor", action="store_true")
    p.add_argument("--post_min_area", type=int, default=30)
    p.add_argument("--post_erode_iters", type=int, default=3)
    p.add_argument("--post_diff_gain", type=float, default=3.0)

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


def _save_overlay_and_binary(
    image: Image.Image,
    output: Dict[str, Any],
    overlay_path: Path,
):
    """
    仅新增可视化输出，不改变模型推理/后处理/评测逻辑。

    输出两类图：
    1) overlay_path：
       原图 + 最终预测分割 mask 的半透明叠加图。
    2) *_binary.png：
       黑色背景 + 白色前景的二值图。

    二值前景由当前模型最终 output["masks"] 中所有实例取并集得到。
    """
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb, dtype=np.uint8)
    h, w = image_np.shape[:2]

    masks = output.get("masks", None)

    if masks is None:
        union_mask = np.zeros((h, w), dtype=bool)
    else:
        masks_t = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
        masks_t = masks_t.detach().cpu()

        if masks_t.numel() == 0:
            union_mask = np.zeros((h, w), dtype=bool)
        else:
            if masks_t.ndim == 4 and masks_t.shape[1] == 1:
                masks_t = masks_t[:, 0]
            elif masks_t.ndim == 2:
                masks_t = masks_t.unsqueeze(0)
            elif masks_t.ndim != 3:
                raise ValueError(
                    f"Unexpected masks shape for visualization: {tuple(masks_t.shape)}"
                )

            # 与原脚本保存 COCO 预测时的逻辑一致：m > 0 作为前景
            masks_np = masks_t.numpy()
            union_mask = np.any(masks_np > 0, axis=0)

            if union_mask.shape != (h, w):
                raise ValueError(
                    f"Mask size {union_mask.shape} does not match image size {(h, w)}"
                )

    # ---------- 1) 原图 + 分割掩膜 ----------
    overlay_np = image_np.copy()

    # 固定绿色半透明 mask；仅影响可视化，不影响任何评测数据
    # 固定黄色半透明 mask
    mask_color = np.zeros_like(overlay_np, dtype=np.uint8)
    mask_color[:] = (0, 0, 255)   # RGB 黄色
    alpha = 0.45

    if np.any(union_mask):
        overlay_np[union_mask] = (
            (1.0 - alpha) * overlay_np[union_mask].astype(np.float32)
            + alpha * mask_color[union_mask].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay_np, mode="RGB").save(overlay_path)

    # ---------- 2) 黑底白前景二值图 ----------
    binary_np = np.zeros((h, w), dtype=np.uint8)
    binary_np[union_mask] = 255

    binary_path = overlay_path.with_name(
        f"{overlay_path.stem}_binary.png"
    )
    Image.fromarray(binary_np, mode="L").save(binary_path)

    print(f"[OK] Saved overlay visualization: {overlay_path}")
    print(f"[OK] Saved binary visualization : {binary_path}")

def _count_instances_from_state(state: Dict[str, Any]) -> int:
    masks = state.get("masks", None)
    if masks is None:
        return 0
    if isinstance(masks, torch.Tensor):
        return int(masks.shape[0])
    return int(np.asarray(masks).shape[0])


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to cpu")
        device = "cpu"

    annot_root = Path(args.annot_root)
    asset_root = Path(args.asset_root)
    result_root = Path(args.result_root)
    cache_root = Path(args.eval_cache_dir) / args.eval_subdir.replace("/", "_")
    cache_root.mkdir(parents=True, exist_ok=True)

    items = build_items(annot_root, asset_root, args.eval_subdir)
    print(f"[INFO] Found {len(items)} annotated images under: {annot_root}/{args.eval_subdir}")

    if args.cleanup_output_dir:
        out_base = result_root if args.eval_subdir.upper() == "ALL" else (result_root / args.eval_subdir)
        if out_base.exists():
            for p in out_base.rglob("*"):
                if p.is_file():
                    p.unlink()
            print(f"[WARN] Cleaned output files under: {out_base}")

    gt = build_gt_coco(items, cat_name=args.single_class_name)
    gt_json_path = cache_root / "gt_coco.json"
    gt_json_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved GT COCO: {gt_json_path}")

    model = _safe_build_sam3(device=device, checkpoint_path=args.checkpoint_path)
    processor = _safe_build_processor(
        model, device=device, confidence_threshold=args.confidence_threshold
    )

    postprocessor = None
    if not args.disable_postprocessor:
        postprocessor = SeedlingPostProcessor(
            min_area=args.post_min_area,
            erode_iters=args.post_erode_iters,
            diff_gain=args.post_diff_gain,
        )
        print("[INFO] postprocessor 已启用")
    else:
        print("[INFO] postprocessor 已禁用")

    pred_results: List[Dict[str, Any]] = []
    for image_id, it in enumerate(items, start=1):
        img_path: Path = it["image_path"]
        rel_dir: Path = it["rel_dir"]
        out_path = (result_root / "7.22" / "zao" / img_path.name).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.text_prompt)
        output = _normalize_model_output(output, image_hw=(h, w), device=device)

        raw_n = _count_instances_from_state(output)

        # ====== 后处理链：SAM3 -> postprocessor ======
        if postprocessor is not None:
            output = postprocessor.process(image, output)
            output = _fix_output_after_postprocess(output, image_hw=(h, w), device=device)
        post_n = _count_instances_from_state(output)

        _save_overlay_and_binary(image, output, out_path)

        masks_t = output["masks"]
        scores_t = output["scores"]

        if masks_t.ndim == 4 and masks_t.shape[1] == 1:
            masks_t = masks_t[:, 0]

        inst = 0
        for i in range(int(masks_t.shape[0])):
            m = masks_t[i].detach().cpu().numpy()
            m = (m > 0).astype(np.uint8)
            if int(m.sum()) == 0:
                continue

            sc = float(scores_t[i].detach().cpu().item()) if i < scores_t.numel() else 1.0
            pred_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": 1,
                    "segmentation": _encode_rle(m),
                    "bbox": _mask_to_bbox(m),
                    "score": float(sc),
                }
            )
            inst += 1

        print(
            f"[OK] {rel_dir}/{img_path.name} -> {out_path} | "
            f"raw={raw_n}, after_post={post_n}, final_saved={inst}"
        )

    pred_json_path = cache_root / "pred_coco.json"
    pred_json_path.write_text(json.dumps(pred_results, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved Pred COCO results: {pred_json_path} (N={len(pred_results)})")

    mask_metrics = run_coco_eval_segmentation(str(gt_json_path), str(pred_json_path))
    bbox_metrics = run_coco_eval_bbox(str(gt_json_path), str(pred_json_path))

    metrics = {
        **mask_metrics,
        **bbox_metrics,
    }

    print("\n========== SAM3 + Postprocessor COCO Metrics ==========")
    print(f"mask mAP (0.50:0.95): {metrics['mask_mAP_50_95']:.4f}")
    print(f"mask AP50          : {metrics['mask_AP50']:.4f}")
    print(f"mask AP75          : {metrics['mask_AP75']:.4f}")
    print("---------------------------------------")
    print(f"bbox mAP (0.50:0.95): {metrics['bbox_mAP_50_95']:.4f}")
    print(f"bbox AP50           : {metrics['bbox_AP50']:.4f}")
    print(f"bbox AP75           : {metrics['bbox_AP75']:.4f}")
    print("=======================================\n")

    (cache_root / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Saved metrics summary: {cache_root / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
