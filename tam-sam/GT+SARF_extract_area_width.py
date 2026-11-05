import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


# ============================================================
# 1. 路径配置
# ============================================================

# mask1：LabelMe 人工标注
ANNOT_ROOT = Path("/root/autodl-tmp/annotations_23")

# mask2：main_MP.py 输出
COCO_DIR = Path("/root/autodl-tmp/plot_results/results_matting/test23/results_postprocess_89/eval_cache_main/ALL")
PRED_COCO_PATH = COCO_DIR / "pred_coco.json"
GT_COCO_PATH = COCO_DIR / "gt_coco.json"

# 最终 CSV
OUTPUT_CSV = Path(
    "/root/autodl-tmp/plot_results/results_matting/test23/extract_area_width/GT+SARF_area_width.csv"
)

# 行列编号参数：
# 与 p_visualization_utils.py 中 _compute_rowcol_ids_xyxy()
# 的默认逻辑保持一致
FIXED_COLS = None
ROW_THRESHOLD_RATIO = 1.0


# ============================================================
# 2. 基础工具
# ============================================================

def normalize_rel_stem(path_like: str) -> str:
    """
    将：
        6.4/zao/position1.png
        6.4/zao/position1.json
    都规范为：
        6.4/zao/position1

    用于 LabelMe JSON 与 COCO image_info 对齐。
    """
    s = str(path_like).replace("\\", "/")
    p = Path(s)
    return str(p.with_suffix("")).replace("\\", "/")


def parse_image_path_fields(file_name: str) -> Tuple[str, str, str]:
    """
    从图像相对路径中解析：
        日期、时段、托盘号

    示例：
        6.4/wan/position1.png
        -> 6.4, wan, position1
    """
    normalized = str(file_name).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    if len(parts) < 3:
        raise ValueError(
            f"图像路径格式不足以解析 日期/时段/托盘号: {file_name}"
        )

    date = parts[-3]
    period = parts[-2]
    tray = Path(parts[-1]).stem

    return date, period, tray


def safe_decode_rle(segmentation: Dict[str, Any]) -> np.ndarray:
    """
    解码 pred_coco.json 中的 COCO RLE。
    兼容 JSON 中 counts 为字符串的情况。
    """
    rle = dict(segmentation)

    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = mask[..., 0]

    return (mask > 0).astype(np.uint8)


# ============================================================
# 3. LabelMe 单个 shape -> mask
#    不合并不同 shape
# ============================================================

def shape_to_mask(
    shape: Dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:

    points = shape.get("points", [])
    if not points:
        return np.zeros((height, width), dtype=np.uint8)

    shape_type = shape.get("shape_type", "polygon")

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if shape_type == "polygon":
        if len(points) < 3:
            return np.zeros((height, width), dtype=np.uint8)

        xy = [(float(x), float(y)) for x, y in points]
        draw.polygon(xy, outline=1, fill=1)

    elif shape_type == "rectangle":
        if len(points) < 2:
            return np.zeros((height, width), dtype=np.uint8)

        (x0, y0), (x1, y1) = points[:2]

        left = min(float(x0), float(x1))
        top = min(float(y0), float(y1))
        right = max(float(x0), float(x1))
        bottom = max(float(y0), float(y1))

        draw.rectangle(
            [left, top, right, bottom],
            outline=1,
            fill=1,
        )

    elif shape_type == "circle":
        if len(points) < 2:
            return np.zeros((height, width), dtype=np.uint8)

        (cx, cy), (px, py) = points[:2]

        radius = float(
            ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        )

        draw.ellipse(
            [
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ],
            outline=1,
            fill=1,
        )

    else:
        # point / line / linestrip 等不作为冠层面状 mask
        return np.zeros((height, width), dtype=np.uint8)

    return np.array(mask_img, dtype=np.uint8)


# ============================================================
# 4. 连通域拆分
#    每一个连通 mask = 一个样本
# ============================================================

def split_connected_components(
    mask: np.ndarray,
) -> List[np.ndarray]:
    """
    不进行不同实例之间的合并。

    仅把当前 mask 自身按 8 邻域拆成独立连通区域。
    每一个非空连通区域都作为一个样本。
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    if int(mask_u8.sum()) == 0:
        return []

    num_labels, labels = cv2.connectedComponents(
        mask_u8,
        connectivity=8,
    )

    components = []

    for label_id in range(1, num_labels):
        comp = (labels == label_id).astype(np.uint8)

        if int(comp.sum()) == 0:
            continue

        components.append(comp)

    return components


# ============================================================
# 5. 冠面积与冠幅
# ============================================================

def compute_area_width(
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    冠面积：
        mask 前景像素数，单位 pixel^2

    冠幅：
        mask 最小外接旋转矩形长轴，
        即 cv2.minAreaRect() 的 max(width, height)，单位 pixel
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    canopy_area = float(mask_u8.sum())

    if canopy_area <= 0:
        return 0.0, 0.0

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contours) == 0:
        return canopy_area, 0.0

    # 当前输入已经是一个连通区域
    cnt = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    rect_w, rect_h = rect[1]

    canopy_width = float(max(rect_w, rect_h))

    return canopy_area, canopy_width


# ============================================================
# 6. mask -> XYXY
# ============================================================

def mask_to_xyxy(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    ]


# ============================================================
# 7. m_n 编号
#    复制 p_visualization_utils.py 的核心逻辑
# ============================================================

def compute_rowcol_ids_xyxy(
    boxes_xyxy,
    fixed_cols=None,
    row_threshold_ratio=1.0,
):
    """
    根据边界框 XYXY 绝对坐标生成 '行_列' 编号 m_n。

    逻辑与 p_visualization_utils.py 保持一致：
    - 先按 y 中心从上到下；
    - 同行按 x 中心从左到右；
    - 自动分行阈值 =
      row_threshold_ratio × 边界框高度中位数；
    - 输出：1_1、1_2、...、2_1、2_2...
    """
    boxes = np.asarray(boxes_xyxy, dtype=float)

    if len(boxes) == 0:
        return {}

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return {
            i: str(i + 1)
            for i in range(len(boxes))
        }

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h = y2 - y1

    # 先按 y，再按 x
    order = np.lexsort((cx, cy))

    ids_map = {}

    if fixed_cols and fixed_cols > 0:
        for rank, orig_idx in enumerate(order):
            row = rank // fixed_cols + 1
            col = rank % fixed_cols + 1
            ids_map[int(orig_idx)] = f"{row}_{col}"
        return ids_map

    median_h = np.median(h) if len(h) else 1.0
    row_thr = max(
        1.0,
        float(row_threshold_ratio) * float(median_h),
    )

    rows = []
    current = []
    last_cy = None

    for orig_idx in order:
        i = int(orig_idx)

        if last_cy is None:
            current = [i]
            last_cy = cy[i]
            continue

        if abs(cy[i] - last_cy) > row_thr:
            current.sort(key=lambda j: cx[j])
            rows.append(current)

            current = [i]
            last_cy = cy[i]
        else:
            current.append(i)

    if current:
        current.sort(key=lambda j: cx[j])
        rows.append(current)

    for row_idx, row_items in enumerate(rows, start=1):
        for col_idx, i in enumerate(row_items, start=1):
            ids_map[int(i)] = f"{row_idx}_{col_idx}"

    return ids_map


# ============================================================
# 8. 获取 LabelMe 图像尺寸
# ============================================================

def get_labelme_size(
    data: Dict[str, Any],
    fallback_w: Optional[int] = None,
    fallback_h: Optional[int] = None,
) -> Tuple[int, int]:

    w = data.get("imageWidth", None)
    h = data.get("imageHeight", None)

    if w is not None and h is not None:
        return int(w), int(h)

    if fallback_w is not None and fallback_h is not None:
        return int(fallback_w), int(fallback_h)

    raise ValueError(
        "LabelMe JSON 中缺少 imageWidth/imageHeight，"
        "且 GT COCO 中也没有可用尺寸。"
    )


# ============================================================
# 9. 读取 mask1：LabelMe GT
# ============================================================

def load_mask1_from_labelme(
    json_path: Path,
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width, height = get_labelme_size(
        data,
        fallback_w=image_w,
        fallback_h=image_h,
    )

    if width != image_w or height != image_h:
        print(
            f"[WARN] LabelMe 尺寸与 COCO 尺寸不同："
            f"{json_path} | "
            f"LabelMe={width}x{height}, "
            f"COCO={image_w}x{image_h}"
        )

    samples = []

    shapes = data.get("shapes", [])

    for shape_index, shape in enumerate(shapes, start=1):
        raw_mask = shape_to_mask(
            shape,
            width=width,
            height=height,
        )

        # 一个 shape 正常对应一个连通幼苗；
        # 若异常存在多个连通块，则按“一个连通 mask 一个样本”拆开
        components = split_connected_components(raw_mask)

        for component_index, comp in enumerate(
            components,
            start=1,
        ):
            samples.append(
                {
                    "mask": comp,
                    "shape_index": shape_index,
                    "component_index": component_index,
                    "label": shape.get("label", ""),
                }
            )

    return samples


# ============================================================
# 10. 读取 mask2：pred_coco.json
# ============================================================

def load_mask2_from_predictions(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    samples = []

    for pred_index, pred in enumerate(
        predictions,
        start=1,
    ):
        segmentation = pred.get("segmentation", None)

        if segmentation is None:
            continue

        raw_mask = safe_decode_rle(segmentation)

        # 同样严格按“一个连通 mask = 一个样本”
        components = split_connected_components(raw_mask)

        for component_index, comp in enumerate(
            components,
            start=1,
        ):
            samples.append(
                {
                    "mask": comp,
                    "pred_index": pred_index,
                    "component_index": component_index,
                    "score": float(pred.get("score", 1.0)),
                }
            )

    return samples


# ============================================================
# 11. mask1 / mask2 匹配
# ============================================================

def pairwise_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> Tuple[int, float]:
    """
    返回：
        intersection_pixels
        IoU
    """
    a = mask1.astype(bool)
    b = mask2.astype(bool)

    intersection = int(
        np.logical_and(a, b).sum()
    )

    if intersection <= 0:
        return 0, 0.0

    union = int(
        np.logical_or(a, b).sum()
    )

    iou = (
        float(intersection / union)
        if union > 0
        else 0.0
    )

    return intersection, iou


def match_mask1_mask2(
    mask1_samples: List[Dict[str, Any]],
    mask2_samples: List[Dict[str, Any]],
) -> Tuple[
    List[Tuple[int, int, int, float]],
    List[int],
    List[int],
]:
    """
    匹配规则：
    - mask1 与 mask2 有像素交集 -> 候选为同一株；
    - 两侧输入都已经是独立连通 mask，
      因此有交集时二者并集也是连通的；
    - 为避免一个 mask 同时匹配多个 mask，
      使用一对一贪心匹配：
        先按交集像素数从大到小，
        再按 IoU 从大到小。

    返回：
        matches:
            [(gt_idx, pred_idx, intersection_px, iou), ...]
        unmatched_gt
        unmatched_pred
    """

    candidates = []

    for gi, gt_sample in enumerate(mask1_samples):
        gt_mask = gt_sample["mask"]

        for pi, pred_sample in enumerate(mask2_samples):
            pred_mask = pred_sample["mask"]

            intersection, iou = pairwise_overlap(
                gt_mask,
                pred_mask,
            )

            if intersection > 0:
                candidates.append(
                    (
                        gi,
                        pi,
                        intersection,
                        iou,
                    )
                )

    # 优先匹配交集最大的区域
    candidates.sort(
        key=lambda x: (
            x[2],
            x[3],
        ),
        reverse=True,
    )

    used_gt = set()
    used_pred = set()
    matches = []

    for gi, pi, intersection, iou in candidates:
        if gi in used_gt:
            continue

        if pi in used_pred:
            continue

        matches.append(
            (
                gi,
                pi,
                intersection,
                iou,
            )
        )

        used_gt.add(gi)
        used_pred.add(pi)

    unmatched_gt = [
        i
        for i in range(len(mask1_samples))
        if i not in used_gt
    ]

    unmatched_pred = [
        i
        for i in range(len(mask2_samples))
        if i not in used_pred
    ]

    return (
        matches,
        unmatched_gt,
        unmatched_pred,
    )


# ============================================================
# 12. 构建统一植株记录
# ============================================================

def build_plant_records(
    mask1_samples: List[Dict[str, Any]],
    mask2_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    (
        matches,
        unmatched_gt,
        unmatched_pred,
    ) = match_mask1_mask2(
        mask1_samples,
        mask2_samples,
    )

    records = []

    # --------------------------------------------------------
    # 已匹配：mask1 + mask2 同一株
    # --------------------------------------------------------
    for (
        gt_idx,
        pred_idx,
        intersection_px,
        match_iou,
    ) in matches:

        gt_sample = mask1_samples[gt_idx]
        pred_sample = mask2_samples[pred_idx]

        gt_mask = gt_sample["mask"]
        pred_mask = pred_sample["mask"]

        # 仅用于确定该“统一植株记录”的空间位置
        # 不参与面积/冠幅计算
        representative_mask = np.logical_or(
            gt_mask > 0,
            pred_mask > 0,
        ).astype(np.uint8)

        records.append(
            {
                "match_status": "GT+AF",
                "gt_sample": gt_sample,
                "pred_sample": pred_sample,
                "intersection_px": intersection_px,
                "match_iou": match_iou,
                "representative_mask": representative_mask,
            }
        )

    # --------------------------------------------------------
    # 只有 GT
    # --------------------------------------------------------
    for gt_idx in unmatched_gt:
        gt_sample = mask1_samples[gt_idx]

        records.append(
            {
                "match_status": "GT_only",
                "gt_sample": gt_sample,
                "pred_sample": None,
                "intersection_px": 0,
                "match_iou": 0.0,
                "representative_mask": gt_sample["mask"],
            }
        )

    # --------------------------------------------------------
    # 只有 AF
    # --------------------------------------------------------
    for pred_idx in unmatched_pred:
        pred_sample = mask2_samples[pred_idx]

        records.append(
            {
                "match_status": "AF_only",
                "gt_sample": None,
                "pred_sample": pred_sample,
                "intersection_px": 0,
                "match_iou": 0.0,
                "representative_mask": pred_sample["mask"],
            }
        )

    return records


# ============================================================
# 13. LabelMe JSON 索引
# ============================================================

def build_labelme_index(
    annot_root: Path,
) -> Dict[str, Path]:

    index = {}

    json_files = sorted(
        [
            p
            for p in annot_root.rglob("*.json")
            if ".ipynb_checkpoints" not in p.parts
            and not p.name.lower().endswith("-checkpoint.json")
        ],
        key=lambda p: str(p).lower(),
    )

    for json_path in json_files:
        rel = json_path.relative_to(annot_root)

        key = normalize_rel_stem(
            str(rel)
        )

        index[key] = json_path

    return index


# ============================================================
# 14. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 检查输入
    # --------------------------------------------------------
    if not ANNOT_ROOT.exists():
        raise FileNotFoundError(
            f"找不到 LabelMe 标注目录：{ANNOT_ROOT}"
        )

    if not GT_COCO_PATH.exists():
        raise FileNotFoundError(
            f"找不到：{GT_COCO_PATH}"
        )

    if not PRED_COCO_PATH.exists():
        raise FileNotFoundError(
            f"找不到：{PRED_COCO_PATH}"
        )

    # --------------------------------------------------------
    # 读取 GT COCO
    # 这里只使用 images 信息完成 image_id、文件名和尺寸对齐
    # 不使用其中 GT segmentation 计算 mask1
    # --------------------------------------------------------
    with open(
        GT_COCO_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        gt_coco = json.load(f)

    images = gt_coco.get(
        "images",
        [],
    )

    # --------------------------------------------------------
    # 读取 MP 预测
    # --------------------------------------------------------
    with open(
        PRED_COCO_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        pred_results = json.load(f)

    if not isinstance(pred_results, list):
        raise ValueError(
            "pred_coco.json 顶层应为 list。"
        )

    pred_by_image_id = {}

    for pred in pred_results:
        image_id = int(pred["image_id"])

        pred_by_image_id.setdefault(
            image_id,
            [],
        ).append(pred)

    # --------------------------------------------------------
    # LabelMe JSON 索引
    # --------------------------------------------------------
    labelme_index = build_labelme_index(
        ANNOT_ROOT
    )

    print(
        f"[INFO] LabelMe JSON 数量："
        f"{len(labelme_index)}"
    )

    print(
        f"[INFO] COCO images 数量："
        f"{len(images)}"
    )

    print(
        f"[INFO] AF 原始预测实例数量："
        f"{len(pred_results)}"
    )

    all_rows = []
    matched_json_keys = set()

    # --------------------------------------------------------
    # 逐图片处理
    # --------------------------------------------------------
    for image_order, image_info in enumerate(
        images,
        start=1,
    ):

        image_id = int(
            image_info["id"]
        )

        file_name = str(
            image_info["file_name"]
        ).replace("\\", "/")

        image_name = Path(
            file_name
        ).name

        date, period, tray = parse_image_path_fields(
            file_name
        )

        image_w = int(
            image_info["width"]
        )

        image_h = int(
            image_info["height"]
        )

        image_key = normalize_rel_stem(
            file_name
        )

        json_path = labelme_index.get(
            image_key,
            None,
        )

        if json_path is not None:
            matched_json_keys.add(
                image_key
            )

            mask1_samples = load_mask1_from_labelme(
                json_path=json_path,
                image_w=image_w,
                image_h=image_h,
            )
        else:
            mask1_samples = []

            print(
                f"[WARN] 未找到对应 LabelMe JSON："
                f"{file_name}"
            )

        image_predictions = pred_by_image_id.get(
            image_id,
            [],
        )

        mask2_samples = load_mask2_from_predictions(
            image_predictions
        )

        print(
            f"[{image_order}/{len(images)}] "
            f"{file_name} | "
            f"mask1={len(mask1_samples)}, "
            f"mask2={len(mask2_samples)}"
        )

        # ----------------------------------------------------
        # GT / AF 匹配成统一植株记录
        # ----------------------------------------------------
        plant_records = build_plant_records(
            mask1_samples,
            mask2_samples,
        )

        # ----------------------------------------------------
        # 每个统一记录只占一个空间位置
        # 再统一进行 m_n 编号
        # ----------------------------------------------------
        representative_boxes = [
            mask_to_xyxy(
                record[
                    "representative_mask"
                ]
            )
            for record in plant_records
        ]

        ids_map = compute_rowcol_ids_xyxy(
            representative_boxes,
            fixed_cols=FIXED_COLS,
            row_threshold_ratio=ROW_THRESHOLD_RATIO,
        )

        # ----------------------------------------------------
        # 提取两种 mask 各自的面积与冠幅
        # ----------------------------------------------------
        for record_idx, record in enumerate(
            plant_records
        ):

            plant_id = ids_map.get(
                record_idx,
                str(record_idx + 1),
            )

            gt_sample = record[
                "gt_sample"
            ]

            pred_sample = record[
                "pred_sample"
            ]

            # mask1 / GT
            if gt_sample is not None:
                gt_area, gt_width = compute_area_width(
                    gt_sample["mask"]
                )

                gt_shape_index = gt_sample.get(
                    "shape_index",
                    None,
                )

                gt_component_index = gt_sample.get(
                    "component_index",
                    None,
                )
            else:
                gt_area = np.nan
                gt_width = np.nan
                gt_shape_index = np.nan
                gt_component_index = np.nan

            # mask2 / MP
            if pred_sample is not None:
                mp_area, mp_width = compute_area_width(
                    pred_sample["mask"]
                )

                mp_score = pred_sample.get(
                    "score",
                    np.nan,
                )

                mp_pred_index = pred_sample.get(
                    "pred_index",
                    None,
                )

                mp_component_index = pred_sample.get(
                    "component_index",
                    None,
                )
            else:
                mp_area = np.nan
                mp_width = np.nan
                mp_score = np.nan
                mp_pred_index = np.nan
                mp_component_index = np.nan

            all_rows.append(
                {
                    "日期": date,
                    "时段": period,
                    "托盘号": tray,
                    "id": plant_id,
                    "match_status": record[
                        "match_status"
                    ],
                    "GT_冠面积": gt_area,
                    "GT_冠幅": gt_width,
                    "AF_冠面积": mp_area,
                    "AF_冠幅": mp_width,
                }
            )

    # --------------------------------------------------------
    # 检查 annotations_89 中是否存在没有出现在 gt_coco images 的 JSON
    # --------------------------------------------------------
    unmatched_json_keys = sorted(
        set(labelme_index.keys())
        - matched_json_keys
    )

    if unmatched_json_keys:
        print(
            f"[WARN] 有 {len(unmatched_json_keys)} 个 LabelMe JSON "
            f"没有在 gt_coco.json 的 images 中找到对应图片。"
        )

        for key in unmatched_json_keys[:20]:
            print(
                f"       unmatched JSON: {labelme_index[key]}"
            )

        if len(unmatched_json_keys) > 20:
            print(
                "       ... 其余未显示"
            )

    # --------------------------------------------------------
    # 保存 CSV
    # --------------------------------------------------------
    columns = [
        "日期",
        "时段",
        "托盘号",
        "id",
        "match_status",
        "GT_冠面积",
        "GT_冠幅",
        "AF_冠面积",
        "AF_冠幅",
    ]

    df = pd.DataFrame(
        all_rows,
        columns=columns,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------
    status_counts = (
        df["match_status"].value_counts()
        if len(df) > 0
        else pd.Series(dtype=int)
    )

    print(
        "\n"
        "==================== 完成 ===================="
    )

    print(
        f"处理图片数量：{len(images)}"
    )

    print(
        f"统一植株记录数：{len(df)}"
    )

    print(
        f"GT+AF 匹配数："
        f"{int(status_counts.get('GT+AF', 0))}"
    )

    print(
        f"GT_only 数："
        f"{int(status_counts.get('GT_only', 0))}"
    )

    print(
        f"AF_only 数："
        f"{int(status_counts.get('AF_only', 0))}"
    )

    print(
        f"结果 CSV：{OUTPUT_CSV}"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
