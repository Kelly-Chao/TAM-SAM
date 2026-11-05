import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as mask_utils


CACHE_ROOT = Path('/root/autodl-tmp/plot_results/results_matting/eval_cache_main/ALL')
GT_JSON_PATH = CACHE_ROOT / 'gt_coco.json'
PRED_JSON_PATH = CACHE_ROOT / 'pred_coco_results.json'
OUT_JSON_PATH = CACHE_ROOT / 'miou_summary.json'
PER_IMAGE_JSON_PATH = CACHE_ROOT / 'miou_per_image.json'


def decode_rle(segmentation: Dict) -> np.ndarray:
    """COCO RLE -> bool mask [H,W]"""
    return mask_utils.decode(segmentation).astype(bool)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def greedy_match_ious(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> List[float]:
    """
    按你的图中逻辑：
    1. 先计算所有 pred-gt 两两 IoU
    2. 每次取当前全局最大 IoU
    3. 该 pred 和 gt 配成一对后都不再参与匹配
    4. 直到不能再匹配
    返回：所有成功匹配对的 IoU 列表
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return []

    iou_mat = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_mat[i, j] = iou(pm, gm)

    matched_ious: List[float] = []
    used_pred = set()
    used_gt = set()

    while True:
        best_i = -1
        best_j = -1
        best_v = -1.0

        for i in range(iou_mat.shape[0]):
            if i in used_pred:
                continue
            for j in range(iou_mat.shape[1]):
                if j in used_gt:
                    continue
                v = float(iou_mat[i, j])
                if v > best_v:
                    best_v = v
                    best_i = i
                    best_j = j

        if best_i == -1 or best_j == -1:
            break

        used_pred.add(best_i)
        used_gt.add(best_j)
        matched_ious.append(best_v)

        if len(used_pred) == iou_mat.shape[0] or len(used_gt) == iou_mat.shape[1]:
            break

    return matched_ious


def group_gt_by_image(gt_data: Dict) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, str]]:
    image_id_to_file = {int(im['id']): im['file_name'] for im in gt_data.get('images', [])}
    grouped: Dict[int, List[np.ndarray]] = {}
    for ann in gt_data.get('annotations', []):
        image_id = int(ann['image_id'])
        grouped.setdefault(image_id, []).append(decode_rle(ann['segmentation']))
    return grouped, image_id_to_file


def group_pred_by_image(pred_data: List[Dict]) -> Dict[int, List[np.ndarray]]:
    grouped: Dict[int, List[np.ndarray]] = {}
    for ann in pred_data:
        image_id = int(ann['image_id'])
        grouped.setdefault(image_id, []).append(decode_rle(ann['segmentation']))
    return grouped


def main():
    if not GT_JSON_PATH.exists():
        raise FileNotFoundError(f'找不到 GT 文件: {GT_JSON_PATH}')
    if not PRED_JSON_PATH.exists():
        raise FileNotFoundError(f'找不到预测文件: {PRED_JSON_PATH}')

    gt_data = json.loads(GT_JSON_PATH.read_text(encoding='utf-8'))
    pred_data = json.loads(PRED_JSON_PATH.read_text(encoding='utf-8'))

    gt_by_image, image_id_to_file = group_gt_by_image(gt_data)
    pred_by_image = group_pred_by_image(pred_data)

    image_ids = sorted(set(image_id_to_file.keys()) | set(gt_by_image.keys()) | set(pred_by_image.keys()))

    per_image_results = []
    image_mious = []

    for image_id in image_ids:
        gt_masks = gt_by_image.get(image_id, [])
        pred_masks = pred_by_image.get(image_id, [])
        matched_ious = greedy_match_ious(pred_masks, gt_masks)

        # 按你图中的定义：
        # 每张图只对“成功匹配到的一对一实例对”的 IoU 求平均
        # 然后再对整个数据集做 overall mean IoU
        if len(matched_ious) == 0:
            image_miou = 0.0
        else:
            image_miou = float(np.mean(matched_ious))

        image_mious.append(image_miou)
        per_image_results.append({
            'image_id': image_id,
            'file_name': image_id_to_file.get(image_id, str(image_id)),
            'num_pred': len(pred_masks),
            'num_gt': len(gt_masks),
            'num_matched': len(matched_ious),
            'matched_ious': [float(x) for x in matched_ious],
            'image_mIoU': image_miou,
        })

    overall_miou = float(np.mean(image_mious)) if len(image_mious) > 0 else 0.0

    summary = {
        'metric_definition': '对每张图先做预测实例与真实实例的两两IoU，随后做贪心一对一匹配；每张图只对成功匹配的实例对IoU取平均；最后对所有图像的image_mIoU再取平均。',
        'num_images': len(image_ids),
        'overall_mIoU': overall_miou,
    }

    OUT_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    PER_IMAGE_JSON_PATH.write_text(json.dumps(per_image_results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n========== mIoU (按你给的图中逻辑) ==========' )
    print(f"图像数             : {summary['num_images']}")
    print(f"overall mIoU      : {summary['overall_mIoU']:.6f}")
    print('=============================================\n')
    print(f'[OK] 已保存: {OUT_JSON_PATH}')
    print(f'[OK] 已保存: {PER_IMAGE_JSON_PATH}')


if __name__ == '__main__':
    main()
