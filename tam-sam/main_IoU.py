import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse  # 新增：参数解析，对齐main_mask_N

import numpy as np
from pycocotools import mask as mask_utils


# =========================
# 路径配置 & 参数解析（对齐main_mask_N）
# =========================
def parse_args():
    p = argparse.ArgumentParser("IoU Evaluation (Annotated Images Only)")
    p.add_argument("--eval_subdir", default="ALL", help='如 "6.4/zao" 或 ALL')
    p.add_argument("--annot_root", default="/root/autodl-tmp/annotations_23")
    p.add_argument("--result_root", default="/root/autodl-tmp/results_MP23")
    p.add_argument("--eval_cache_dir", default="/root/autodl-tmp/results_MP23/outputs")
    return p.parse_args()

args = parse_args()

# 根路径
ROOT = Path("/root/sam3-main")

# 对齐main_mask_N的路径配置
ANNOT_ROOT = Path(args.annot_root)
RESULT_ROOT = Path(args.result_root)
# 计算cache_root（与main_mask_N完全一致）
CACHE_ROOT = Path(args.eval_cache_dir) / args.eval_subdir.replace("/", "_")

# GT和Pred路径（与main_mask_N的输出路径完全对齐）

# Follow the same cache path as main_MP.py. For eval_subdir=ALL this resolves
# to <eval_cache_dir>/ALL/gt_coco.json and pred_coco.json; for e.g. 6.28/wan
# it resolves to <eval_cache_dir>/6.28_wan/.
GT_JSON_PATH = CACHE_ROOT / "gt_coco.json"
PRED_JSON_PATH = CACHE_ROOT / "pred_coco.json"


RESULT_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def normalize_relpath(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def collect_annotated_relpaths(annot_root: Path) -> Set[str]:
    """Return annotation-relative paths without suffix; supports PNG/JPG/TIF."""
    relpaths = set()
    for jp in annot_root.rglob("*.json"):
        if ".ipynb_checkpoints" in jp.parts or jp.name.lower().endswith("-checkpoint.json"):
            continue
        rel_stem = jp.relative_to(annot_root).with_suffix("")
        relpaths.add(normalize_relpath(str(rel_stem)))
    return relpaths


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_index(gt_coco: dict):
    """
    返回：
    - image_id_to_relfile: image_id -> 6.28/zao/position1.png
    - relfile_to_image_id: 6.28/zao/position1.png -> image_id
    - gt_anns_by_image: image_id -> [ann, ...]
    """
    image_id_to_relfile: Dict[int, str] = {}
    relfile_to_image_id: Dict[str, int] = {}
    gt_anns_by_image: Dict[int, List[dict]] = {}

    for img in gt_coco.get("images", []):
        image_id = int(img["id"])
        rel_file = normalize_relpath(img["file_name"])
        image_id_to_relfile[image_id] = rel_file
        relfile_to_image_id[rel_file] = image_id

    for ann in gt_coco.get("annotations", []):
        image_id = int(ann["image_id"])
        gt_anns_by_image.setdefault(image_id, []).append(ann)

    return image_id_to_relfile, relfile_to_image_id, gt_anns_by_image


def build_pred_index(pred_coco: list):
    """
    pred_coco_results.json 是 list，每项至少含：
    image_id / segmentation / score / bbox
    """
    pred_anns_by_image: Dict[int, List[dict]] = {}
    for ann in pred_coco:
        image_id = int(ann["image_id"])
        pred_anns_by_image.setdefault(image_id, []).append(ann)
    return pred_anns_by_image


def decode_segmentation(segmentation) -> np.ndarray:
    """
    COCO RLE -> 0/1 mask, shape=(H, W)
    """
    segmentation = dict(segmentation)
    if isinstance(segmentation.get("counts"), str):
        segmentation["counts"] = segmentation["counts"].encode("ascii")
    mask = mask_utils.decode(segmentation)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def build_iou_matrix(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]) -> np.ndarray:
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)

    mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            if gm.shape != pm.shape:
                raise ValueError(f"mask shape mismatch: gt={gm.shape}, pred={pm.shape}")
            mat[i, j] = compute_iou(gm, pm)
    return mat


def greedy_one_to_one_match(iou_mat: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    一对一最佳匹配（贪心）：
    - 先取全局最大 IoU
    - 对应 gt 和 pred 都不再重复使用
    """
    matches: List[Tuple[int, int, float]] = []
    if iou_mat.size == 0:
        return matches

    used_gt = set()
    used_pred = set()

    pairs = []
    rows, cols = iou_mat.shape
    for i in range(rows):
        for j in range(cols):
            pairs.append((float(iou_mat[i, j]), i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matches.append((i, j, iou))

    return matches


def try_find_result_file(result_root: Path, rel_png: str) -> Path:
    """
    只用于检查 results 下对应文件是否存在
    """
    p = result_root / rel_png
    if p.exists():
        return p

    stem_path = Path(rel_png)
    stem = stem_path.stem
    parent = stem_path.parent
    for ext in RESULT_EXTS:
        cand = result_root / parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    return p


def main():
    print("========== IoU Evaluation (Annotated Images Only) ==========")

    # 1) 路径检查
    if not ANNOT_ROOT.exists():
        print(f"[ERROR] annotations 目录不存在: {ANNOT_ROOT}")
        return
    if not GT_JSON_PATH.exists():
        print(f"[ERROR] gt_coco.json 不存在: {GT_JSON_PATH}")
        return
    if not PRED_JSON_PATH.exists():
        print(f"[ERROR] pred_coco_results.json 不存在: {PRED_JSON_PATH}")
        return

    # 2) 收集已标注图像
    annotated_relpaths = collect_annotated_relpaths(ANNOT_ROOT)
    print(f"[INFO] 已标注图像数量（按 annotations/*.json 统计）: {len(annotated_relpaths)}")

    # 3) 读取 COCO 文件
    gt_coco = load_json(GT_JSON_PATH)
    pred_coco = load_json(PRED_JSON_PATH)

    image_id_to_relfile, relfile_to_image_id, gt_anns_by_image = build_gt_index(gt_coco)
    pred_anns_by_image = build_pred_index(pred_coco)

    print(f"[INFO] gt_coco.json 中 images 数量: {len(image_id_to_relfile)}")
    print(f"[INFO] pred_coco_results.json 中实例数量: {len(pred_coco)}")

    # 4) 通过相对路径 stem 匹配标注与 COCO 图像，兼容 PNG/JPG/TIF 等后缀。
    gt_relfiles = set(relfile_to_image_id.keys())
    gt_stems = {normalize_relpath(str(Path(rp).with_suffix(""))): rp for rp in gt_relfiles}
    valid_relpaths = sorted(
        gt_stems[stem] for stem in annotated_relpaths if stem in gt_stems
    )

    if valid_relpaths:
        print("[OK] annotations 与 gt_coco 的相对路径匹配")
    else:
        print("[ERROR] annotations 与 gt_coco 没有匹配到任何相对路径")
        return

    annot_not_in_gt = sorted(annotated_relpaths - set(gt_stems.keys()))
    if annot_not_in_gt:
        print(f"[WARN] 有 {len(annot_not_in_gt)} 张已标注图像未出现在 gt_coco.json 中，已忽略")  # 【修改点2】：对齐run_IoU文案
    # 【删除原main_IoU的else分支】：run_IoU无此提示

    # 5) 保留全部已标注图像，包括没有任何预测实例的图像，避免漏检图像被静默排除。
    final_relpaths = valid_relpaths
    valid_image_ids = [relfile_to_image_id[rp] for rp in valid_relpaths]
    image_ids_with_pred = {img_id for img_id in valid_image_ids if img_id in pred_anns_by_image}
    no_pred_relpaths = [rp for rp in valid_relpaths if relfile_to_image_id[rp] not in image_ids_with_pred]

    print(f"[INFO] 参与 IoU 计算/报告的图像数: {len(final_relpaths)}")
    print(f"[INFO] 已标注但没有任何预测实例的图像数: {len(no_pred_relpaths)}")  # 【修改点2】：对齐run_IoU文案

    # 6) 逐图计算 IoU
    per_image_results = []
    all_matched_ious = []
    total_gt_instances = 0
    total_pred_instances = 0
    total_matched_pairs = 0
    # 【修改点3】：删除shape_mismatch_found变量（run_IoU无此全局标记）

    for relpath in final_relpaths:
        image_id = relfile_to_image_id[relpath]
        gt_anns = gt_anns_by_image.get(image_id, [])
        pred_anns = pred_anns_by_image.get(image_id, [])

        gt_masks = [decode_segmentation(a["segmentation"]) for a in gt_anns]
        pred_masks = [decode_segmentation(a["segmentation"]) for a in pred_anns]

        total_gt_instances += len(gt_masks)
        total_pred_instances += len(pred_masks)

        # 尺寸检查
        shape_ok = True
        if gt_masks and pred_masks:
            gshape = gt_masks[0].shape
            for pm in pred_masks:
                if pm.shape != gshape:
                    shape_ok = False
                    break

        if not shape_ok:
            print(f"[ERROR] mask 尺寸不匹配: {relpath}")  # 【修改点2】：对齐run_IoU文案
            continue  # run_IoU仅打印后continue，无全局标记

        iou_mat = build_iou_matrix(gt_masks, pred_masks)
        matches = greedy_one_to_one_match(iou_mat)

        matched_ious = [m[2] for m in matches]
        image_mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        total_matched_pairs += len(matches)
        all_matched_ious.extend(matched_ious)

        result_file = try_find_result_file(RESULT_ROOT, relpath)

        per_image_results.append({
            "file_name": relpath,
            "image_id": image_id,
            "result_file_exists": result_file.exists(),  # 【修改点4】：字段名对齐run_IoU（result_exists→result_file_exists）
            "gt_instances": len(gt_masks),
            "pred_instances": len(pred_masks),
            "matched_pairs": len(matches),
            "mean_iou": image_mean_iou,
            "matched_ious": matched_ious,
        })

    # 【删除原main_IoU的shape_mismatch_found判断输出】：run_IoU无此逻辑

    # 7) results 路径检查
    result_missing_count = sum(0 if x["result_file_exists"] else 1 for x in per_image_results)  # 同步字段名修改

    # 8) 汇总 & 排查结果输出（【修改点5】：对齐run_IoU的“排查结果”板块）
    overall_mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    print("\n==================== 排查结果 ====================")  # 【修改点2】：对齐run_IoU文案
    print("annotations -> gt_coco 文件名匹配 ✅")
    print("仅使用已标注图像进行 IoU 计算 ✅")
    if no_pred_relpaths:
        print("部分已标注图像没有预测实例；这些图像已纳入逐图报告，IoU 记为 0（没有匹配对）⚠️")
    else:
        print("所有已标注图像都有预测结果记录 ✅")

    if result_missing_count == 0:
        print("results 路径检查通过（对应结果图文件存在）✅")
    else:
        print(f"有 {result_missing_count} 张图在 results 目录下未找到对应文件 ⚠️")

    print("\n==================== IoU 统计 ====================")  # 【修改点2】：对齐run_IoU文案
    print(f"参与计算图像数          : {len(per_image_results)}")
    print(f"GT 实例总数             : {total_gt_instances}")
    print(f"Pred 实例总数           : {total_pred_instances}")
    print(f"成功匹配实例对数        : {total_matched_pairs}")
    print(f"总体 Mean IoU           : {overall_mean_iou:.6f}")

    # 【删除原main_IoU的IoU阈值判断输出】：run_IoU无此逻辑

    print("\n==================== 每张图结果 ====================")
    for item in per_image_results:
        print(
            f"{item['file_name']} | "
            f"GT={item['gt_instances']} | Pred={item['pred_instances']} | "  # 【修改点2】：对齐run_IoU文案（Pred大写）
            f"Matched={item['matched_pairs']} | IoU={item['mean_iou']:.6f}"
        )

    # 9) 保存详细结果（【修改点6】：字段名对齐run_IoU）
    out_path = CACHE_ROOT / "iou_from_annotated_only_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "annot_root": str(ANNOT_ROOT),
            "result_root": str(RESULT_ROOT),
            "gt_coco_path": str(GT_JSON_PATH),  # 【修改点4】：gt_json→gt_coco_path
            "pred_coco_path": str(PRED_JSON_PATH),  # 【修改点4】：pred_json→pred_coco_path
            "annotated_image_count": len(annotated_relpaths),
            "valid_gt_image_count": len(valid_relpaths),
            "final_eval_image_count": len(per_image_results),
            "images_without_pred_count": len(no_pred_relpaths),
            "images_without_pred": no_pred_relpaths,
            "total_gt_instances": total_gt_instances,
            "total_pred_instances": total_pred_instances,
            "total_matched_pairs": total_matched_pairs,
            "overall_mean_iou": overall_mean_iou,
            "per_image_results": per_image_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 详细结果已保存到: {out_path}")
    print("===================================================")  # 【修改点2】：对齐run_IoU文案


if __name__ == "__main__":
    main()
