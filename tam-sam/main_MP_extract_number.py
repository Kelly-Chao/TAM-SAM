import argparse
import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.e_mp_visualization_utils import plot_results, extract_features_and_save_csv

# ====== 2个后处理模块导入 ======
from sam3.postprocessor import SeedlingPostProcessor
from sam3.trimapsegment import TrimapSegmenter, TrimapSegmentConfig


ASSET_ROOT_DEFAULT = "/root/sam3-main/assets_64zao1"
RESULT_ROOT_DEFAULT = "/root/sam3-main/results_MP_64zao1"
CHECKPOINT_PATH_DEFAULT = "/root/autodl-tmp/sam3.pt"
MORNING_EXCEL_PATH_DEFAULT = "/root/sam3-main/早上数据.xls"
FEATURE_CSV_PATH_DEFAULT = "/root/autodl-tmp/results_MP_64zao1/MP_features_64zao1.csv"
MISSING_FAMILY_LOG_PATH_DEFAULT = None
AUTO_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def parse_args():
    p = argparse.ArgumentParser(
        "newSAM3 batch inference + trimapsegment + postprocessor + visualization + feature CSV"
    )

    # ====== 数据与输出路径 ======
    p.add_argument(
        "--asset_root",
        default=ASSET_ROOT_DEFAULT,
        help="图像根目录；默认只处理 /root/sam3-main/all_assets/zao下的图像",
    )
    p.add_argument(
        "--eval_subdir",
        default="ALL",
        help='要处理 asset_root 下的哪个子目录；ALL 表示递归处理全部日期目录，例如 6.4/position1.png、6.16/position1.png 等',
    )
    p.add_argument("--result_root", default=RESULT_ROOT_DEFAULT, help="可视化结果保存根目录")
    p.add_argument("--checkpoint_path", default=CHECKPOINT_PATH_DEFAULT)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ====== 幼苗 / 植物盆检测提示词与置信度阈值 ======
    p.add_argument(
        "--text_prompt",
        default="pine seedling leaves, needle leaves",
        help="用于检测幼苗/叶片目标的文本提示词",
    )
    p.add_argument(
        "--pot_text_prompt",
        default="plant pot, flower pot, seedling pot",
        help="用于检测植物盆的文本提示词；检测到的 pot box 用于 m_n 行列编号",
    )
    p.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.40,
        help="幼苗/叶片目标检测置信度阈值",
    )
    p.add_argument(
        "--pot_confidence_threshold",
        type=float,
        default=0.8,
        help="植物盆检测置信度阈值",
    )

    # ====== 植物盆检测与行列编号参数 ======
    p.add_argument(
        "--disable_pot_detection",
        action="store_true",
        help="若启用：不额外检测植物盆；编号将退化为根据植物目标框排序",
    )
    p.add_argument(
        "--row_threshold",
        type=float,
        default=250.0,
        help="植物盆按 y1 分行的阈值；当前盆 y1 与当前行第一个盆 y1 的差值 <= 该值则算同一行",
    )
    p.add_argument(
        "--min_plant_pot_overlap_pixels",
        type=int,
        default=1,
        help="植株 mask 与某个盆框至少重合多少个像素，才认为该植株属于该盆；默认 1，表示必须有实际重合",
    )

    # ====== 特征 CSV 参数 ======
    p.add_argument("--feature_csv_path", default=FEATURE_CSV_PATH_DEFAULT)
    p.add_argument("--morning_excel_path", default=MORNING_EXCEL_PATH_DEFAULT)
    p.add_argument(
        "--missing_family_log_path",
        default=MISSING_FAMILY_LOG_PATH_DEFAULT,
        help="找不到家系号的盆/植物编号记录文件",
    )
    p.add_argument(
        "--overwrite_feature_csv",
        action="store_true",
        help="若启用：运行前删除已有特征 CSV，重新生成",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="若启用：跳过已经在特征表或处理日志中存在的图像，只处理未完成图像",
    )
    p.add_argument(
        "--resume_mode",
        default="both",
        choices=["table", "log", "both"],
        help="断点续跑判断方式：table=查特征表，log=查已处理日志，both=两者任一命中就跳过",
    )
    p.add_argument(
        "--processed_log_path",
        default=None,
        help="已处理图像日志路径；默认保存在 feature_csv_path 同目录下 processed_images.txt",
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="从第几张图开始跑；例如上次停在 533/693，可设 --start_index 534",
    )
    p.add_argument(
        "--skip_existing_min_values",
        type=int,
        default=1,
        help="按特征表续跑时，某图像对应日期列至少有多少个非空值才认为已处理",
    )
    p.add_argument(
        "--fixed_cols",
        type=int,
        default=None,
        help="若每个托盘每行固定列数，可填写该值；默认 None，沿用可视化文件中的自动分行编号规则",
    )

    # ====== 输出控制 ======
    p.add_argument(
        "--cleanup_output_dir",
        action="store_true",
        help="若启用：先清空 result_root 下对应输出文件夹中的旧可视化图（谨慎）",
    )
    p.add_argument(
        "--no_visualization",
        action="store_true",
        help="若启用：不保存带 mask/bbox/编号的可视化图，只提取特征 CSV",
    )
    p.add_argument(
        "--hide_pot_boxes",
        action="store_true",
        help="若启用：可视化图中不绘制植物盆 box；默认绘制植物盆 box 和 m_n 编号",
    )

    # ====== trimapsegment 参数 ======
    p.add_argument("--disable_trimapsegment", action="store_true")
    p.add_argument("--trimap_erode_iters", type=int, default=5)
    p.add_argument("--trimap_dilate_iters", type=int, default=5)
    p.add_argument("--trimap_kernel_size", type=int, default=3)
    p.add_argument("--trimap_alpha_threshold", type=float, default=0.5)
    p.add_argument(
        "--fba_weights",
        default="/root/sam3-main/sam3/third_party/FBA_Matting/FBA.pth",
    )
    p.add_argument(
        "--fba_repo_dir",
        default="/root/sam3-main/sam3/third_party/FBA_Matting",
    )
    p.add_argument("--fba_scale", type=float, default=1.0)

    # ====== postprocessor 参数 ======
    p.add_argument("--disable_postprocessor", action="store_true")
    p.add_argument("--post_min_area", type=int, default=30)
    p.add_argument("--post_erode_iters", type=int, default=3)
    p.add_argument("--post_diff_gain", type=float, default=3.0)

    return p.parse_args()


# ===================== 通用辅助函数 =====================
def _natural_key(s: str) -> Tuple:
    parts = []
    for tok in re.split(r"(\d+)", s):
        if tok.isdigit():
            parts.append(int(tok))
        else:
            parts.append(tok.lower())
    return tuple(parts)


def _is_image_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.name.startswith("."):
        return False
    # 只处理文件名包含 position12.png 的图像
    if "position1.png" not in p.name:
        return False
    return p.suffix.lower() in AUTO_EXTS


def build_image_items(asset_root: Path, eval_subdir: str) -> List[Dict[str, Any]]:
    asset_root = asset_root.resolve()
    base = asset_root if eval_subdir.upper() == "ALL" else (asset_root / eval_subdir)
    if not base.exists():
        raise FileNotFoundError(f"Image folder not found: {base}")

    image_files = [p for p in base.rglob("*") if _is_image_file(p)]
    image_files = sorted(image_files, key=lambda x: _natural_key(str(x.relative_to(asset_root))))

    items: List[Dict[str, Any]] = []
    seen = set()
    for img in image_files:
        key = str(img.resolve())
        if key in seen:
            continue
        seen.add(key)
        rel = img.relative_to(asset_root)
        items.append(
            {
                "image_path": img,
                "rel_dir": rel.parent,
            }
        )
    return items


def _masks_to_xyxy_boxes(masks_t: torch.Tensor) -> torch.Tensor:
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


def _safe_build_sam3(device: str, checkpoint_path: str):
    import inspect
    sig = inspect.signature(build_sam3_image_model)
    kwargs = {}
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "checkpoint_path" in sig.parameters:
        kwargs["checkpoint_path"] = checkpoint_path

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


def _filter_output_by_score(output: Dict[str, Any], min_score: float) -> Dict[str, Any]:
    if output is None or "scores" not in output or output["scores"] is None:
        return output

    scores = output["scores"]
    scores_t = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)
    scores_t = scores_t.flatten().float()
    n = int(scores_t.numel())
    if n == 0:
        return output

    keep = scores_t >= float(min_score)
    keep_cpu = keep.detach().cpu().numpy().astype(bool)
    keep_indices = np.where(keep_cpu)[0].tolist()

    filtered = dict(output)
    for key in ["masks", "scores", "boxes", "masks_logits"]:
        value = filtered.get(key, None)
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            if value.shape[0] == n:
                filtered[key] = value[keep.to(value.device)]
        else:
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == n:
                filtered[key] = arr[keep_cpu]
            elif isinstance(value, (list, tuple)) and len(value) == n:
                filtered[key] = [value[i] for i in keep_indices]

    return filtered


def _filter_seedlings_without_pot_overlap(
    output: Dict[str, Any],
    pot_data: Dict[str, Any],
    min_overlap_pixels: int = 1,
) -> Dict[str, Any]:
    if output is None:
        return output

    masks = output.get("masks", None)
    if masks is None:
        return output

    masks_t = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
    if masks_t.ndim == 2:
        masks_t = masks_t.unsqueeze(0).unsqueeze(0)
    elif masks_t.ndim == 3:
        masks_t = masks_t.unsqueeze(1)
    elif masks_t.ndim != 4:
        return output

    n = int(masks_t.shape[0])
    if n == 0:
        return output

    pot_boxes = None
    for key in ["pot_boxes", "pots", "all_pots", "pot_bboxes", "pot_bbox", "plant_pot_boxes"]:
        if key in pot_data and pot_data[key] is not None:
            pot_boxes = pot_data[key]
            break

    if pot_boxes is None:
        keep = torch.zeros((n,), dtype=torch.bool, device=masks_t.device)
    else:
        pot_boxes_t = pot_boxes if isinstance(pot_boxes, torch.Tensor) else torch.as_tensor(pot_boxes)
        pot_boxes_t = pot_boxes_t.detach().to(masks_t.device).float()
        if pot_boxes_t.ndim != 2 or pot_boxes_t.shape[-1] != 4 or pot_boxes_t.shape[0] == 0:
            keep = torch.zeros((n,), dtype=torch.bool, device=masks_t.device)
        else:
            _, _, h, w = masks_t.shape
            keep_list = []
            masks_bool = masks_t[:, 0] > 0
            for i in range(n):
                has_pot = False
                for box in pot_boxes_t:
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, int(np.floor(x1)))
                    y1 = max(0, int(np.floor(y1)))
                    x2 = min(w, int(np.ceil(x2)))
                    y2 = min(h, int(np.ceil(y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    overlap_pixels = int(masks_bool[i, y1:y2, x1:x2].sum().item())
                    if overlap_pixels >= int(min_overlap_pixels):
                        has_pot = True
                        break
                keep_list.append(has_pot)
            keep = torch.as_tensor(keep_list, dtype=torch.bool, device=masks_t.device)

    filtered = dict(output)
    for key in ["masks", "scores", "boxes", "masks_logits"]:
        value = filtered.get(key, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.shape[0] == n:
                filtered[key] = value[keep.to(value.device)]
        else:
            arr = np.asarray(value)
            keep_cpu = keep.detach().cpu().numpy().astype(bool)
            if arr.ndim >= 1 and arr.shape[0] == n:
                filtered[key] = arr[keep_cpu]
            elif isinstance(value, (list, tuple)) and len(value) == n:
                filtered[key] = [v for v, k in zip(value, keep_cpu) if k]

    return filtered

def _safe_detect_pots(
    processor,
    image: Image.Image,
    image_hw: Tuple[int, int],
    device: str,
    pot_text_prompt: str,
    pot_confidence_threshold: float = 0.40,
) -> Tuple[Dict[str, Any], int]:
    try:
        pot_state = processor.set_image(image)
        pot_output = processor.set_text_prompt(state=pot_state, prompt=pot_text_prompt)
        pot_output = _normalize_model_output(pot_output, image_hw=image_hw, device=device)
        pot_output = _filter_output_by_score(pot_output, pot_confidence_threshold)

        pot_n = _count_instances_from_state(pot_output)
        pot_data = {
            "pot_boxes": pot_output.get("boxes"),
            "pot_scores": pot_output.get("scores"),
            "pot_masks": pot_output.get("masks"),
        }
        return pot_data, pot_n
    except Exception as e:
        print(f"[WARN] 植物盆检测失败，将退化为按植物目标框编号: {e}")
        return {}, 0


def _safe_plot(
    image: Image.Image,
    output: Dict[str, Any],
    out_path: Path,
    fixed_cols: Optional[int] = None,
    row_threshold: float = 250.0,
    draw_pot_boxes: bool = True,
):
    try:
        plot_results(
            image,
            output,
            save_name=str(out_path),
            fixed_cols=fixed_cols,
            row_threshold=row_threshold,
            draw_pot_boxes=draw_pot_boxes,
        )
    except Exception as e:
        print(f"[WARN] plot_results 失败，改为保存原图: {e}")
        image.save(out_path)


def _count_instances_from_state(state: Dict[str, Any]) -> int:
    masks = state.get("masks", None)
    if masks is None:
        return 0
    if isinstance(masks, torch.Tensor):
        return int(masks.shape[0])
    return int(np.asarray(masks).shape[0])


def _normalize_tray_id_for_resume(x):
    s = "" if x is None else str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = Path(s).stem
    m = re.search(r"position\s*0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"position{int(m.group(1))}"
    return s


def _infer_date_from_image_path_for_resume(image_path):
    parts = [str(p) for p in Path(str(image_path)).parts]
    candidates = []
    for p in parts:
        if re.fullmatch(r"\d{1,2}\.\d{1,2}", p):
            candidates.append(p)
    if candidates:
        return candidates[-1]
    m = re.search(r"(\d{1,2}\.\d{1,2})", str(image_path))
    return m.group(1) if m else ""


def _read_feature_table_for_resume(feature_csv_path: Path):
    feature_csv_path = Path(feature_csv_path)
    candidate_paths = []
    candidate_paths.append(feature_csv_path)
    if feature_csv_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        candidate_paths.append(feature_csv_path.with_suffix(".xlsx"))

    table_path = None
    for p in candidate_paths:
        if p.exists():
            table_path = p
            break
    if table_path is None:
        return None

    try:
        if table_path.suffix.lower() in {".xlsx", ".xlsm"}:
            df = pd.read_excel(table_path, dtype=object)
            for col in ["托盘号", "编号", "家系号", "组别"]:
                if col in df.columns:
                    df[col] = df[col].ffill()
        else:
            # ✅ 修复编码错误：自动识别编码，不再强制 utf-8-sig
            df = pd.read_csv(table_path, dtype=object, encoding="gbk", errors="ignore")
    except Exception as e:
        try:
            df = pd.read_csv(table_path, dtype=object, encoding="gb2312", errors="ignore")
        except:
            df = pd.read_csv(table_path, dtype=object, encoding="latin-1", errors="ignore")

    if "托盘号" not in df.columns:
        return None
    df["托盘号"] = df["托盘号"].map(_normalize_tray_id_for_resume)
    return df


def _image_has_existing_feature_data(feature_df, image_path, min_values=1):
    if feature_df is None or len(feature_df) == 0:
        return False
    tray_id = _normalize_tray_id_for_resume(image_path)
    date_col = _infer_date_from_image_path_for_resume(image_path)
    if not tray_id or not date_col or date_col not in feature_df.columns:
        return False

    sub = feature_df.loc[feature_df["托盘号"] == tray_id, date_col]
    if len(sub) == 0:
        return False
    non_empty = sub.notna() & (sub.astype(str).str.strip() != "")
    return int(non_empty.sum()) >= int(min_values)


def _load_processed_images(log_path: Path):
    if not log_path.exists():
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def _mark_image_processed(log_path: Path, image_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(Path(image_path).resolve()) + "\n")


@contextlib.contextmanager
def _suppress_terminal_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    asset_root = Path(args.asset_root).resolve()
    result_root = Path(args.result_root).resolve()
    feature_csv_path = Path(args.feature_csv_path).resolve()

    if args.processed_log_path is None:
        processed_log_path = feature_csv_path.parent / "processed_images.txt"
    else:
        processed_log_path = Path(args.processed_log_path).resolve()

    resume_feature_df = None
    processed_images = set()
    if args.resume:
        if args.resume_mode in {"table", "both"}:
            resume_feature_df = _read_feature_table_for_resume(feature_csv_path)
        if args.resume_mode in {"log", "both"}:
            processed_images = _load_processed_images(processed_log_path)

    items = build_image_items(asset_root, args.eval_subdir)
    if args.cleanup_output_dir and result_root.exists():
        shutil.rmtree(result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite_feature_csv and feature_csv_path.exists():
        feature_csv_path.unlink()
    if args.overwrite_feature_csv and args.missing_family_log_path:
        missing_family_log_path = Path(args.missing_family_log_path).resolve()
        if missing_family_log_path.exists():
            missing_family_log_path.unlink()

    with _suppress_terminal_output():
        model = _safe_build_sam3(device=device, checkpoint_path=args.checkpoint_path)
        processor = _safe_build_processor(
            model, device=device, confidence_threshold=args.confidence_threshold
        )
        pot_processor = None
        if not args.disable_pot_detection:
            pot_processor = _safe_build_processor(
                model, device=device, confidence_threshold=args.pot_confidence_threshold
            )

        trimap_segmenter = None
        if not args.disable_trimapsegment:
            trimap_cfg = TrimapSegmentConfig(
                erode_iters=args.trimap_erode_iters,
                dilate_iters=args.trimap_dilate_iters,
                kernel_size=args.trimap_kernel_size,
                alpha_threshold=args.trimap_alpha_threshold,
                fba_weights=args.fba_weights,
                fba_repo_dir=args.fba_repo_dir,
                fba_scale=args.fba_scale,
            )
            trimap_segmenter = TrimapSegmenter(config=trimap_cfg, device=device)
        else:
            trimap_segmenter = None

        postprocessor = None
        if not args.disable_postprocessor:
            postprocessor = SeedlingPostProcessor(
                min_area=args.post_min_area,
                erode_iters=args.post_erode_iters,
                diff_gain=args.post_diff_gain,
            )
        else:
            postprocessor = None

    for image_id, it in enumerate(items, start=1):
        img_path: Path = it["image_path"]

        if image_id < args.start_index:
            continue

        img_abs = str(img_path.resolve())
        if args.resume:
            skip_by_log = args.resume_mode in {"log", "both"} and img_abs in processed_images
            skip_by_table = (
                args.resume_mode in {"table", "both"}
                and _image_has_existing_feature_data(
                    resume_feature_df,
                    img_path,
                    min_values=args.skip_existing_min_values,
                )
            )
            if skip_by_log or skip_by_table:
                continue

        rel_dir: Path = it["rel_dir"]
        out_path = (result_root / rel_dir / img_path.name).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{image_id}/{len(items)}] {img_path}", flush=True)

        with _suppress_terminal_output():
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.text_prompt)
            output = _normalize_model_output(output, image_hw=(h, w), device=device)

            if trimap_segmenter is not None:
                output = trimap_segmenter.process(image, output)
                output = _fix_output_after_postprocess(output, image_hw=(h, w), device=device)

            if postprocessor is not None:
                output = postprocessor.process(image, output)
                output = _fix_output_after_postprocess(output, image_hw=(h, w), device=device)

            if not args.disable_pot_detection:
                pot_data, _ = _safe_detect_pots(
                    processor=pot_processor if pot_processor is not None else processor,
                    image=image,
                    image_hw=(h, w),
                    device=device,
                    pot_text_prompt=args.pot_text_prompt,
                    pot_confidence_threshold=args.pot_confidence_threshold,
                )
                output = _filter_seedlings_without_pot_overlap(
                    output,
                    pot_data,
                    min_overlap_pixels=args.min_plant_pot_overlap_pixels,
                )
                if pot_data:
                    output.update(pot_data)

            if not args.no_visualization:
                _safe_plot(
                    image,
                    output,
                    out_path,
                    fixed_cols=args.fixed_cols,
                    row_threshold=args.row_threshold,
                    draw_pot_boxes=not args.hide_pot_boxes,
                )

            extract_features_and_save_csv(
                img_pil=image,
                results=output,
                csv_path=str(feature_csv_path),
                image_path=str(img_path),
                excel_path=args.morning_excel_path,
                fixed_cols=args.fixed_cols,
                row_threshold=args.row_threshold,
                image_root=str(asset_root),
                strict_image_root=True,
                missing_family_log_path=args.missing_family_log_path,
            )

            if args.resume:
                _mark_image_processed(processed_log_path, img_path)


if __name__ == "__main__":
    main()
