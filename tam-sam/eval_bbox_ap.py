import argparse
import json
from pathlib import Path
from typing import Dict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise RuntimeError("缺少 pycocotools：pip install -U pycocotools") from e


def run_coco_eval_bbox(gt_json_path: str, pred_json_path: str) -> Dict[str, float]:
    coco_gt = COCO(gt_json_path)
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])

    preds = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))
    if not isinstance(preds, list) or len(preds) == 0:
        return {
            "bbox_mAP_50_95": 0.0,
            "bbox_AP50": 0.0,
            "bbox_AP75": 0.0,
        }

    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "bbox_mAP_50_95": float(coco_eval.stats[0]),
        "bbox_AP50": float(coco_eval.stats[1]),
        "bbox_AP75": float(coco_eval.stats[2]),
    }


def parse_args():
    p = argparse.ArgumentParser("独立脚本：根据 gt_coco.json 和 pred_coco.json 计算 bbox AP50 / AP75 / mAP")
    p.add_argument("--gt_json", required=True, help="GT COCO JSON 路径")
    p.add_argument("--pred_json", required=True, help="预测 COCO JSON 路径")
    p.add_argument("--save_json", default=None, help="可选：将结果保存为 JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_coco_eval_bbox(args.gt_json, args.pred_json)

    print("\n========== BBox Detection Metrics ==========")
    print(f"bbox mAP (0.50:0.95): {metrics['bbox_mAP_50_95']:.4f}")
    print(f"bbox AP50          : {metrics['bbox_AP50']:.4f}")
    print(f"bbox AP75          : {metrics['bbox_AP75']:.4f}")
    print("============================================\n")

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] 已保存结果到: {args.save_json}")
