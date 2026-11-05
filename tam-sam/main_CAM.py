#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Original SAM3 multi-instance Grad-CAM.

For one image, every SAM3 query that passes the same confidence threshold as
Sam3Processor is explained independently. Each per-instance CAM is independently
min-max normalized, then all instance CAMs are fused by pixel-wise MAX.
No predicted-mask gating is applied to the CAM itself.

Default target layer:
    backbone_out["backbone_fpn"][-1]

Why this layer:
    In the provided Sam3Image implementation, num_feature_levels=1 and
    _get_img_feats() uses the last backbone_fpn level as the visual feature
    entering the transformer encoder.

Outputs for each image:
    *_SAM3_overlay.png
    *_binary.png
    *_CAM.png
    *_CAM_fixed.png
    *_CAM_heatmap.png
    *_CAM.npy
    *_CAM_meta.json
    *_CAM_panel.png
    instances/<stem>/*_CAM.png / *_CAM.npy / *_mask.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate real gradient-based Grad-CAM maps for the SAM3 visual backbone."
    )
    p.add_argument(
        "--input_path",
        default="/root/sam3-main/assets_p1/6.4/zao/position1.png",
        help="Single image path or a directory. Directories are searched recursively.",
    )
    p.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/plot_results/results/CAM_SAM3/6.4/zao",
        help="Root directory for CAM outputs.",
    )
    p.add_argument(
        "--checkpoint_path",
        default="/root/autodl-tmp/sam3.pt",
        help="SAM3 checkpoint path.",
    )
    p.add_argument(
        "--text_prompt",
        default="pine seedling leaves, needle leaves",
        help="Text prompt used by SAM3.",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.45,
        help="Same confidence threshold used by main_MP.py.",
    )
    p.add_argument(
        "--cam_target",
        default="mask",
        choices=["mask", "joint"],
        help=(
            "mask: mask-probability target for each instance; "
            "joint: mask target multiplied by the SAM3 joint score."
        ),
    )
    p.add_argument(
        "--instance",
        default="all",
        help=(
            'Which prediction(s) to explain: "top1", "all", or a zero-based query index, '
            'for example --instance 3.'
        ),
    )
    p.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.60,
        help="Maximum opacity of the adaptive CAM overlay.",
    )
    p.add_argument(
        "--fixed_alpha",
        type=float,
        default=0.45,
        help="Fixed opacity for paper-style cross-model comparison.",
    )
    p.add_argument(
        "--seg_alpha",
        type=float,
        default=0.45,
        help="Opacity of the blue SAM3 segmentation overlay.",
    )
    p.add_argument(
        "--no_instance_files",
        action="store_true",
        help="Do not save per-instance CAM/heatmap/mask/npy files.",
    )
    p.add_argument(
        "--no_cam_relu",
        action="store_true",
        help="Disable the standard Grad-CAM ReLU step.",
    )
    p.add_argument(
        "--no_panel",
        action="store_true",
        help="Do not save the 4-column paper-style panel.",
    )
    return p.parse_args()


def build_model_and_processor(checkpoint_path: str, device: str, confidence_threshold: float):
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    model.eval()

    # Grad-CAM only needs d(target)/d(target_feature), not parameter gradients.
    # Freezing parameters reduces memory usage substantially.
    for p in model.parameters():
        p.requires_grad_(False)

    processor = Sam3Processor(
        model,
        resolution=1008,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    return model, processor


def preprocess_image(image: Image.Image, processor: Sam3Processor, device: str) -> torch.Tensor:
    """Reproduce Sam3Processor.set_image() preprocessing without inference_mode."""
    x = v2.functional.to_image(image).to(device)
    x = processor.transform(x).unsqueeze(0)
    return x


def detach_nested(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: detach_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_nested(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_nested(v) for v in obj)
    return obj


def prepare_backbone_for_cam(
    model,
    processor: Sam3Processor,
    image_tensor: torch.Tensor,
    text_prompt: str,
):
    """
    Compute image/text backbone outputs without storing the full ViT graph, then
    re-enable gradients only for backbone_fpn[-1]. This is sufficient for
    Grad-CAM at that selected activation layer.
    """
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(image_tensor)
        text_out = model.backbone.forward_text([text_prompt], device=processor.device)

    backbone_out = detach_nested(backbone_out)
    text_out = detach_nested(text_out)
    backbone_out.update(text_out)

    if "backbone_fpn" not in backbone_out:
        raise RuntimeError(
            "SAM3 backbone output has no 'backbone_fpn'. "
            "The model structure differs from the supplied implementation."
        )

    fpn = list(backbone_out["backbone_fpn"])
    if len(fpn) == 0:
        raise RuntimeError("backbone_fpn is empty.")

    target_feature = fpn[-1].detach().requires_grad_(True)
    fpn[-1] = target_feature
    backbone_out["backbone_fpn"] = fpn
    return backbone_out, target_feature


def forward_grounding_with_grad(model, processor: Sam3Processor, backbone_out: Dict):
    """
    Reproduce the differentiable part of Sam3Processor.set_text_prompt() /
    _forward_grounding() without @torch.inference_mode().
    """
    geometric_prompt = model._get_dummy_prompt()
    with torch.enable_grad():
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
    return outputs


def compute_joint_scores(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Match the supplied Sam3Processor score definition:
        sigmoid(pred_logits) * sigmoid(presence_logit_dec)
    Returns [Q] for the first/only image.
    """
    if "pred_logits" not in outputs:
        raise RuntimeError("SAM3 output missing 'pred_logits'.")

    score_prob = outputs["pred_logits"].sigmoid()

    if "presence_logit_dec" in outputs and outputs["presence_logit_dec"] is not None:
        presence_prob = outputs["presence_logit_dec"].sigmoid()
        while presence_prob.ndim < score_prob.ndim:
            presence_prob = presence_prob.unsqueeze(1)
        score_prob = score_prob * presence_prob

    if score_prob.ndim >= 1 and score_prob.shape[-1] == 1:
        score_prob = score_prob.squeeze(-1)
    if score_prob.ndim == 2:
        return score_prob[0]
    if score_prob.ndim == 1:
        return score_prob

    raise RuntimeError(f"Unexpected score tensor shape: {tuple(score_prob.shape)}")


def parse_instance_selection(instance_arg: str, scores: torch.Tensor, threshold: float) -> List[int]:
    q = int(scores.numel())
    if q == 0:
        raise RuntimeError("SAM3 produced zero queries.")

    arg = str(instance_arg).strip().lower()
    if arg == "top1":
        return [int(torch.argmax(scores.detach()).item())]

    if arg == "all":
        keep = torch.where(scores.detach() > threshold)[0].tolist()
        if len(keep) == 0:
            keep = [int(torch.argmax(scores.detach()).item())]
            print(
                "[WARN] No query exceeded confidence threshold; "
                "falling back to the top-scoring query for CAM."
            )
        return [int(i) for i in keep]

    try:
        idx = int(arg)
    except ValueError as e:
        raise ValueError('--instance must be "top1", "all", or an integer query index.') from e

    if idx < 0 or idx >= q:
        raise IndexError(f"Query index {idx} is out of range [0, {q - 1}].")
    return [idx]


def build_instance_cam_target(
    raw_query_mask_logit: torch.Tensor,
    final_instance_mask: torch.Tensor,
    query_score: torch.Tensor,
    cam_target: str,
) -> torch.Tensor:
    """
    Build one scalar target for one SAM3 instance.

    To match the TAF-SAM3 CAM implementation, the final SAM3 mask is used only
    as a DETACHED spatial support for the differentiable raw SAM3 mask logit.
    The CAM itself is never multiplied by the predicted mask.
    """
    if final_instance_mask.ndim == 3 and final_instance_mask.shape[0] == 1:
        final_instance_mask = final_instance_mask[0]

    support = final_instance_mask.detach().float()[None, None]
    support_small = F.interpolate(
        support,
        size=raw_query_mask_logit.shape[-2:],
        mode="nearest",
    )[0, 0] > 0.5

    mask_prob = raw_query_mask_logit.sigmoid()
    if bool(support_small.any()):
        mask_term = mask_prob[support_small].mean()
    else:
        raw_fg = mask_prob.detach() > 0.5
        mask_term = mask_prob[raw_fg].mean() if bool(raw_fg.any()) else mask_prob.mean()

    if cam_target == "mask":
        return mask_term
    if cam_target == "joint":
        return mask_term * query_score
    raise ValueError(f"Unsupported cam_target: {cam_target}")


def gradcam_one_instance(
    target_scalar: torch.Tensor,
    target_feature: torch.Tensor,
    use_relu: bool,
    retain_graph: bool,
):
    """Standard Grad-CAM for exactly one SAM3 instance/query."""
    grad = torch.autograd.grad(
        outputs=target_scalar,
        inputs=target_feature,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]

    if target_feature.ndim != 4:
        raise RuntimeError(
            f"Expected target feature [B,C,H,W], got {tuple(target_feature.shape)}."
        )

    weights = grad.mean(dim=(2, 3), keepdim=True)
    raw_cam = (weights * target_feature).sum(dim=1)[0]
    if use_relu:
        raw_cam = torch.relu(raw_cam)

    raw_min = float(raw_cam.min().detach().cpu())
    raw_max = float(raw_cam.max().detach().cpu())
    raw_std = float(raw_cam.std().detach().cpu())
    grad_abs_mean = float(grad.abs().mean().detach().cpu())
    grad_abs_max = float(grad.abs().max().detach().cpu())

    cam = raw_cam.detach().float().cpu().numpy()
    if raw_max - raw_min < 1e-12:
        cam01 = np.zeros_like(cam, dtype=np.float32)
        constant = True
    else:
        cam01 = ((cam - raw_min) / (raw_max - raw_min)).astype(np.float32)
        constant = False

    return cam01, {
        "grad_abs_mean": grad_abs_mean,
        "grad_abs_max": grad_abs_max,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_std": raw_std,
        "constant": constant,
    }


def normalize_query_masks(outputs: Dict[str, torch.Tensor], q_count: int) -> torch.Tensor:
    """Normalize pred_masks to [Q,Hm,Wm]."""
    pred_masks = outputs["pred_masks"]
    if pred_masks.ndim == 3 and pred_masks.shape[0] == q_count:
        return pred_masks
    if pred_masks.ndim == 4:
        if pred_masks.shape[0] == 1 and pred_masks.shape[1] == q_count:
            return pred_masks[0]
        if pred_masks.shape[0] == q_count and pred_masks.shape[1] == 1:
            return pred_masks[:, 0]
    raise RuntimeError(
        f"Cannot normalize pred_masks={tuple(pred_masks.shape)} to [Q,H,W], Q={q_count}."
    )


def final_sam3_instance_masks(
    query_masks: torch.Tensor,
    selected_queries: List[int],
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Match Sam3Processor exactly:
        pred_masks -> bilinear resize to original size -> sigmoid -> >0.5
    Returns [N,1,H,W] bool.
    """
    if len(selected_queries) == 0:
        return torch.zeros((0, 1, height, width), dtype=torch.bool, device=query_masks.device)

    logits = torch.stack([query_masks[q] for q in selected_queries], dim=0).unsqueeze(1)
    probs = F.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()
    return probs > 0.5


def resize_cam(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.clip(
        cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC),
        0.0,
        1.0,
    ).astype(np.float32)


def cam_to_heatmap_rgb(cam01: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(cam01 * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def blend_fixed(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (
        (1.0 - alpha) * image_rgb.astype(np.float32)
        + alpha * heatmap_rgb.astype(np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_adaptive(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    cam01: np.ndarray,
    max_alpha: float,
) -> np.ndarray:
    """Visualization-only adaptive opacity; low-CAM background stays close to input."""
    alpha_map = (np.clip(cam01, 0.0, 1.0) * float(np.clip(max_alpha, 0.0, 1.0)))[..., None]
    out = (
        image_rgb.astype(np.float32) * (1.0 - alpha_map)
        + heatmap_rgb.astype(np.float32) * alpha_map
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def make_union_binary(masks: torch.Tensor, width: int, height: int) -> np.ndarray:
    if masks.numel() == 0:
        return np.zeros((height, width), dtype=np.uint8)
    union = masks[:, 0].detach().cpu().numpy().astype(bool).any(axis=0)
    return union.astype(np.uint8) * 255


def make_sam3_overlay(
    image_rgb: np.ndarray,
    masks: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    """Original image + all SAM3 masks, fixed semi-transparent blue."""
    out = image_rgb.copy()
    if masks.numel() == 0:
        return out
    union = masks[:, 0].detach().cpu().numpy().astype(bool).any(axis=0)
    blue = np.zeros_like(image_rgb, dtype=np.uint8)
    blue[..., 2] = 255
    if union.any():
        out[union] = (
            (1.0 - alpha) * out[union].astype(np.float32)
            + alpha * blue[union].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
    return out


def save_panel(
    input_rgb: np.ndarray,
    sam3_overlay_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(input_rgb)
    axes[0].set_title("Input", fontsize=12)
    axes[1].imshow(sam3_overlay_rgb)
    axes[1].set_title("SAM3", fontsize=12)
    axes[2].imshow(heatmap_rgb)
    axes[2].set_title("Multi-instance Grad-CAM", fontsize=12)
    axes[3].imshow(overlay_rgb)
    axes[3].set_title("CAM Overlay", fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_images(input_path: Path) -> Tuple[List[Path], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path], input_path.parent

    if input_path.is_dir():
        images = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            raise FileNotFoundError(f"No images found under: {input_path}")
        return images, input_path

    raise FileNotFoundError(f"Input path not found: {input_path}")


def process_one_image(
    image_path: Path,
    relative_root: Path,
    output_root: Path,
    model,
    processor,
    args,
):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_rgb = np.asarray(image, dtype=np.uint8)

    image_tensor = preprocess_image(image, processor, args.device)
    backbone_out, target_feature = prepare_backbone_for_cam(
        model=model,
        processor=processor,
        image_tensor=image_tensor,
        text_prompt=args.text_prompt,
    )

    outputs = forward_grounding_with_grad(
        model=model,
        processor=processor,
        backbone_out=backbone_out,
    )

    scores = compute_joint_scores(outputs)
    selected_queries = parse_instance_selection(
        instance_arg=args.instance,
        scores=scores,
        threshold=args.confidence_threshold,
    )

    # For the intended multi-instance use, default --instance all keeps exactly
    # the same score>threshold queries as Sam3Processor.
    if str(args.instance).strip().lower() == "all":
        selected_queries = [
            int(q)
            for q in torch.where(scores.detach() > args.confidence_threshold)[0].cpu().tolist()
        ]

    q_masks = normalize_query_masks(outputs, q_count=int(scores.numel()))
    final_masks = final_sam3_instance_masks(
        query_masks=q_masks,
        selected_queries=selected_queries,
        width=width,
        height=height,
    )

    try:
        rel_parent = image_path.parent.relative_to(relative_root)
    except ValueError:
        rel_parent = Path()

    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    sam3_overlay = make_sam3_overlay(
        image_rgb=image_rgb,
        masks=final_masks,
        alpha=args.seg_alpha,
    )
    binary = make_union_binary(final_masks, width=width, height=height)
    Image.fromarray(sam3_overlay).save(out_dir / f"{stem}_SAM3_overlay.png")
    Image.fromarray(binary).save(out_dir / f"{stem}_binary.png")

    print(f"[CAM] feature={tuple(target_feature.shape)}")
    print(f"[CAM] selected queries={selected_queries}")

    # No SAM3 instance passed confidence threshold.
    if len(selected_queries) == 0:
        final_cam = np.zeros((height, width), dtype=np.float32)
        heatmap_rgb = cam_to_heatmap_rgb(final_cam)
        Image.fromarray(image_rgb).save(out_dir / f"{stem}_CAM.png")
        Image.fromarray(blend_fixed(image_rgb, heatmap_rgb, args.fixed_alpha)).save(
            out_dir / f"{stem}_CAM_fixed.png"
        )
        Image.fromarray(heatmap_rgb).save(out_dir / f"{stem}_CAM_heatmap.png")
        np.save(out_dir / f"{stem}_CAM.npy", final_cam)
        meta = {
            "image": str(image_path),
            "model": "SAM3",
            "num_instances": 0,
            "confidence_threshold": args.confidence_threshold,
            "cam_method": "instance-wise Grad-CAM",
        }
        (out_dir / f"{stem}_CAM_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[CAM][WARN] no query > {args.confidence_threshold}")
        return

    final_cam = np.zeros((height, width), dtype=np.float32)
    instance_records = []
    instance_outputs = []

    for local_idx, query_idx in enumerate(selected_queries):
        target_scalar = build_instance_cam_target(
            raw_query_mask_logit=q_masks[query_idx],
            final_instance_mask=final_masks[local_idx],
            query_score=scores[query_idx],
            cam_target=args.cam_target,
        )

        cam_small, debug = gradcam_one_instance(
            target_scalar=target_scalar,
            target_feature=target_feature,
            use_relu=not args.no_cam_relu,
            retain_graph=(local_idx < len(selected_queries) - 1),
        )

        # Important for exact TAF-SAM3 comparability:
        # independent min-max happens BEFORE resize, same as TAF script.
        instance_cam = resize_cam(cam_small, width=width, height=height)
        final_cam = np.maximum(final_cam, instance_cam)

        score = float(scores[query_idx].detach().cpu())
        area = int(final_masks[local_idx].detach().sum().cpu())
        rec = {
            "instance_id": local_idx + 1,
            "sam3_query_idx": int(query_idx),
            "sam3_joint_score": score,
            "mask_area": area,
            "grad_abs_mean": debug["grad_abs_mean"],
            "grad_abs_max": debug["grad_abs_max"],
            "cam_raw_min": debug["raw_min"],
            "cam_raw_max": debug["raw_max"],
            "cam_raw_std": debug["raw_std"],
            "cam_constant": debug["constant"],
            "cam_max_after_resize": float(instance_cam.max()),
        }
        instance_records.append(rec)
        instance_outputs.append((rec, instance_cam, final_masks[local_idx]))

        print(
            f"[CAM][{local_idx + 1:03d}] query={query_idx:03d} "
            f"score={score:.4f} area={area} "
            f"grad={debug['grad_abs_mean']:.3e} CAMmax={instance_cam.max():.4f}"
        )

    final_cam = np.clip(final_cam, 0.0, 1.0).astype(np.float32)
    heatmap_rgb = cam_to_heatmap_rgb(final_cam)
    overlay_rgb = blend_adaptive(
        image_rgb=image_rgb,
        heatmap_rgb=heatmap_rgb,
        cam01=final_cam,
        max_alpha=args.overlay_alpha,
    )
    fixed_rgb = blend_fixed(image_rgb, heatmap_rgb, args.fixed_alpha)

    # Requested image-level outputs.
    Image.fromarray(overlay_rgb).save(out_dir / f"{stem}_CAM.png")
    Image.fromarray(fixed_rgb).save(out_dir / f"{stem}_CAM_fixed.png")
    Image.fromarray(heatmap_rgb).save(out_dir / f"{stem}_CAM_heatmap.png")
    np.save(out_dir / f"{stem}_CAM.npy", final_cam)

    if not args.no_instance_files:
        instance_dir = out_dir / "instances" / stem
        instance_dir.mkdir(parents=True, exist_ok=True)
        for rec, instance_cam, instance_mask in instance_outputs:
            prefix = f"instance_{rec['instance_id']:03d}_query{rec['sam3_query_idx']:03d}"
            inst_heatmap = cam_to_heatmap_rgb(instance_cam)
            inst_overlay = blend_adaptive(
                image_rgb=image_rgb,
                heatmap_rgb=inst_heatmap,
                cam01=instance_cam,
                max_alpha=args.overlay_alpha,
            )
            mask_u8 = instance_mask[0].detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(inst_overlay).save(instance_dir / f"{prefix}_CAM.png")
            Image.fromarray(inst_heatmap).save(instance_dir / f"{prefix}_heatmap.png")
            Image.fromarray(mask_u8).save(instance_dir / f"{prefix}_mask.png")
            np.save(instance_dir / f"{prefix}_CAM.npy", instance_cam.astype(np.float32))

    if not args.no_panel:
        save_panel(
            input_rgb=image_rgb,
            sam3_overlay_rgb=sam3_overlay,
            heatmap_rgb=heatmap_rgb,
            overlay_rgb=overlay_rgb,
            out_path=out_dir / f"{stem}_CAM_panel.png",
        )

    meta = {
        "image": str(image_path),
        "model": "SAM3",
        "model_class": type(model).__name__,
        "text_prompt": args.text_prompt,
        "confidence_threshold": args.confidence_threshold,
        "cam_method": "instance-wise Grad-CAM",
        "cam_target": args.cam_target,
        "target_layer": 'backbone_out["backbone_fpn"][-1]',
        "target_feature_shape": list(target_feature.shape),
        "selected_queries": selected_queries,
        "num_instances": len(selected_queries),
        "instance_normalization": "independent min-max",
        "image_level_fusion": "pixel-wise MAX",
        "cam_mask_gating": False,
        "mask_role": "detached spatial support for each instance scalar target only",
        "adaptive_overlay_alpha": args.overlay_alpha,
        "fixed_overlay_alpha": args.fixed_alpha,
        "instances": instance_records,
    }
    (out_dir / f"{stem}_CAM_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[CAM][OK] overlay = {out_dir / (stem + '_CAM.png')}")
    print(f"[CAM][OK] fixed   = {out_dir / (stem + '_CAM_fixed.png')}")
    print(f"[CAM][OK] npy     = {out_dir / (stem + '_CAM.npy')}")

    del outputs, backbone_out, target_feature, image_tensor
    if args.device == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU.")
        args.device = "cpu"

    input_path = Path(args.input_path).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images, relative_root = collect_images(input_path)
    print(f"[INFO] Found {len(images)} image(s).")
    print(f"[INFO] Building SAM3 from: {args.checkpoint_path}")

    model, processor = build_model_and_processor(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    print(
        "[INFO] Grad-CAM target layer: backbone_out['backbone_fpn'][-1]\n"
        f"[INFO] CAM target mode      : {args.cam_target}\n"
        f"[INFO] Instance selection   : {args.instance}\n"
        f"[INFO] Text prompt          : {args.text_prompt}"
    )

    for idx, image_path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] Processing: {image_path}")
        process_one_image(
            image_path=image_path,
            relative_root=relative_root,
            output_root=output_root,
            model=model,
            processor=processor,
            args=args,
        )

    print(f"\n[DONE] All CAM results saved under: {output_root}")


if __name__ == "__main__":
    main()
