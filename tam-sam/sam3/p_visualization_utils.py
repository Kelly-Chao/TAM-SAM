# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import json
import os
import subprocess
from pathlib import Path
import math

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
import torch
from matplotlib.colors import to_rgb
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(lab)
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)


def show_img_tensor(img_batch, vis_img_idx=0):
    MEAN_IMG = np.array([0.5, 0.5, 0.5])
    STD_IMG = np.array([0.5, 0.5, 0.5])
    im_tensor = img_batch[vis_img_idx].detach().cpu()
    assert im_tensor.dim() == 3
    im_tensor = im_tensor.numpy().transpose((1, 2, 0))
    im_tensor = (im_tensor * STD_IMG) + MEAN_IMG
    im_tensor = np.clip(im_tensor, 0, 1)
    plt.imshow(im_tensor)


def draw_box_on_image(image, box, color=(0, 255, 0)):
    image = image.convert("RGB")
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    pixels = image.load()

    # Draw the top and bottom edges
    for i in range(x, x + w):
        pixels[i, y] = color
        pixels[i, y + h - 1] = color
        pixels[i, y + 1] = color
        pixels[i, y + h] = color
        pixels[i, y - 1] = color
        pixels[i, y + h - 2] = color

    # Draw the left and right edges
    for j in range(y, y + h):
        pixels[x, j] = color
        pixels[x + 1, j] = color
        pixels[x - 1, j] = color
        pixels[x + w - 1, j] = color
        pixels[x + w, j] = color
        pixels[x + w - 2, j] = color

    return image


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )


def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def normalize_bbox(bbox_xywh, img_w, img_h):
    if isinstance(bbox_xywh, list):
        assert len(bbox_xywh) == 4
        normalized_bbox = bbox_xywh.copy()
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
    else:
        assert isinstance(bbox_xywh, torch.Tensor)
        normalized_bbox = bbox_xywh.clone()
        normalized_bbox[..., 0] /= img_w
        normalized_bbox[..., 1] /= img_h
        normalized_bbox[..., 2] /= img_w
        normalized_bbox[..., 3] /= img_h
    return normalized_bbox


def visualize_frame_output(frame_idx, video_frames, outputs, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    plt.title(f"frame {frame_idx}")
    img = load_frame(video_frames[frame_idx])
    img_H, img_W, _ = img.shape
    plt.imshow(img)
    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        prob = outputs["out_probs"][i]
        obj_id = outputs["out_obj_ids"][i]
        binary_mask = outputs["out_binary_masks"][i]
        color = COLORS[obj_id % len(COLORS)]
        plot_bbox(
            img_H,
            img_W,
            box_xywh,
            text=f"(id={obj_id}, {prob=:.2f})",
            box_format="XYWH",
            color=color,
        )
        plot_mask(binary_mask, color=color)


def visualize_formatted_frame_output(
    frame_idx,
    video_frames,
    outputs_list,
    titles=None,
    points_list=None,
    points_labels_list=None,
    figsize=(12, 8),
    title_suffix="",
    prompt_info=None,
):
    if isinstance(outputs_list, dict) and frame_idx in outputs_list:
        outputs_list = [outputs_list]
    elif isinstance(outputs_list, dict) and not any(
        isinstance(k, int) for k in outputs_list.keys()
    ):
        single_frame_outputs = {frame_idx: outputs_list}
        outputs_list = [single_frame_outputs]

    num_outputs = len(outputs_list)
    if titles is None:
        titles = [f"Set {i+1}" for i in range(num_outputs)]
    assert len(titles) == num_outputs

    _, axes = plt.subplots(1, num_outputs, figsize=figsize)
    if num_outputs == 1:
        axes = [axes]

    img = load_frame(video_frames[frame_idx])
    img_H, img_W, _ = img.shape

    for idx in range(num_outputs):
        ax, outputs_set, ax_title = axes[idx], outputs_list[idx], titles[idx]
        ax.set_title(f"Frame {frame_idx} - {ax_title}{title_suffix}")
        ax.imshow(img)

        if frame_idx in outputs_set:
            _outputs = outputs_set[frame_idx]
        else:
            print(f"Warning: Frame {frame_idx} not found in outputs_set")
            continue

        if prompt_info and frame_idx == 0:
            if "boxes" in prompt_info:
                for box in prompt_info["boxes"]:
                    x, y, w, h = box
                    plot_bbox(
                        img_H,
                        img_W,
                        [x, y, x + w, y + h],
                        box_format="XYXY",
                        relative_coords=True,
                        color="yellow",
                        linestyle="dashed",
                        text="PROMPT BOX",
                        ax=ax,
                    )

            if "points" in prompt_info and "point_labels" in prompt_info:
                points = np.array(prompt_info["points"])
                labels = np.array(prompt_info["point_labels"])
                points_pixel = points * np.array([img_W, img_H])

                pos_points = points_pixel[labels == 1]
                if len(pos_points) > 0:
                    ax.scatter(
                        pos_points[:, 0],
                        pos_points[:, 1],
                        color="lime",
                        marker="*",
                        s=200,
                        edgecolor="white",
                        linewidth=2,
                        label="Positive Points",
                        zorder=10,
                    )

                neg_points = points_pixel[labels == 0]
                if len(neg_points) > 0:
                    ax.scatter(
                        neg_points[:, 0],
                        neg_points[:, 1],
                        color="red",
                        marker="*",
                        s=200,
                        edgecolor="white",
                        linewidth=2,
                        label="Negative Points",
                        zorder=10,
                    )

        objects_drawn = 0
        for obj_id, binary_mask in _outputs.items():
            mask_sum = (
                binary_mask.sum()
                if hasattr(binary_mask, "sum")
                else np.sum(binary_mask)
            )

            if mask_sum > 0:
                if not isinstance(binary_mask, torch.Tensor):
                    binary_mask = torch.tensor(binary_mask)

                if binary_mask.any():
                    box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
                    box_xyxy = normalize_bbox(box_xyxy, img_W, img_H)
                else:
                    box_xyxy = [0.45, 0.45, 0.55, 0.55]

                color = COLORS[obj_id % len(COLORS)]

                plot_bbox(
                    img_H,
                    img_W,
                    box_xyxy,
                    text=f"(id={obj_id})",
                    box_format="XYXY",
                    color=color,
                    ax=ax,
                )

                mask_np = (
                    binary_mask.numpy()
                    if isinstance(binary_mask, torch.Tensor)
                    else binary_mask
                )
                plot_mask(mask_np, color=color, ax=ax)
                objects_drawn += 1

        if objects_drawn == 0:
            ax.text(
                0.5,
                0.5,
                "No objects detected",
                transform=ax.transAxes,
                fontsize=16,
                ha="center",
                va="center",
                color="red",
                weight="bold",
            )

        if points_list is not None and points_list[idx] is not None:
            show_points(
                points_list[idx], points_labels_list[idx], ax=ax, marker_size=200
            )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def render_masklet_frame(img, outputs, frame_idx=None, alpha=0.5):
    if img.dtype == np.float32 or img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img = img[..., :3]
    height, width = img.shape[:2]
    overlay = img.copy()

    for i in range(len(outputs["out_probs"])):
        obj_id = outputs["out_obj_ids"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = (color * 255).astype(np.uint8)
        mask = outputs["out_binary_masks"][i]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)

    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        obj_id = outputs["out_obj_ids"][i]
        prob = outputs["out_probs"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = tuple(int(x * 255) for x in color)
        x, y, w, h = box_xywh
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)
        if prob is not None:
            label = f"id={obj_id}, p={prob:.2f}"
        else:
            label = f"id={obj_id}"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color255,
            1,
            cv2.LINE_AA,
        )

    if frame_idx is not None:
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def save_masklet_video(video_frames, outputs, out_path, alpha=0.5, fps=10):
    first_img = load_frame(video_frames[0])
    height, width = first_img.shape[:2]
    if first_img.dtype == np.float32 or first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("temp.mp4", fourcc, fps, (width, height))

    outputs_list = [
        (video_frames[frame_idx], frame_idx, outputs[frame_idx])
        for frame_idx in sorted(outputs.keys())
    ]

    for frame, frame_idx, frame_outputs in tqdm(outputs_list):
        img = load_frame(frame)
        overlay = render_masklet_frame(
            img, frame_outputs, frame_idx=frame_idx, alpha=alpha
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    writer.release()
    subprocess.run(["ffmpeg", "-y", "-i", "temp.mp4", out_path])
    print(f"Re-encoded video saved to {out_path}")
    os.remove("temp.mp4")


def save_masklet_image(frame, outputs, out_path, alpha=0.5, frame_idx=None):
    img = load_frame(frame)
    overlay = render_masklet_frame(img, outputs, frame_idx=frame_idx, alpha=alpha)

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = RESULTS_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(overlay).save(str(out_path))
    print(f"Overlay image saved to {out_path}")


def prepare_masks_for_visualization(frame_to_output):
    for frame_idx, out in frame_to_output.items():
        _processed_out = {}
        for idx, obj_id in enumerate(out["out_obj_ids"].tolist()):
            if out["out_binary_masks"][idx].any():
                _processed_out[obj_id] = out["out_binary_masks"][idx]
        frame_to_output[frame_idx] = _processed_out
    return frame_to_output


def convert_coco_to_masklet_format(
    annotations, img_info, is_prediction=False, score_threshold=0.5
):
    outputs = {
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_obj_ids": [],
        "out_binary_masks": [],
    }

    img_h, img_w = img_info["height"], img_info["width"]

    for idx, ann in enumerate(annotations):
        if "bbox" in ann:
            bbox = ann["bbox"]
            if max(bbox) > 1.0:
                bbox = [
                    bbox[0] / img_w,
                    bbox[1] / img_h,
                    bbox[2] / img_w,
                    bbox[3] / img_h,
                ]
        else:
            mask = mask_utils.decode(ann["segmentation"])
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox = [
                    cmin / img_w,
                    rmin / img_h,
                    (cmax - cmin + 1) / img_w,
                    (rmax - rmin + 1) / img_h,
                ]
            else:
                bbox = [0, 0, 0, 0]

        outputs["out_boxes_xywh"].append(bbox)

        if is_prediction:
            prob = ann["score"]
        else:
            prob = 1.0
        outputs["out_probs"].append(prob)

        outputs["out_obj_ids"].append(idx)
        mask = mask_utils.decode(ann["segmentation"])
        mask = (mask > score_threshold).astype(np.uint8)
        outputs["out_binary_masks"].append(mask)

    return outputs


def save_side_by_side_visualization(img, gt_anns, pred_anns, noun_phrase):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    main_title = f"Noun phrase: '{noun_phrase}'"
    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    gt_overlay = render_masklet_frame(img, gt_anns, alpha=0.5)
    ax1.imshow(gt_overlay)
    ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
    ax1.axis("off")

    pred_overlay = render_masklet_frame(img, pred_anns, alpha=0.5)
    ax2.imshow(pred_overlay)
    ax2.set_title("Predictions", fontsize=14, fontweight="bold")
    ax2.axis("off")

    plt.subplots_adjust(top=0.88)
    plt.tight_layout()


def bitget(val, idx):
    return (val >> idx) & 1


def pascal_color_map():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bitget(ind, channel) << shift
        ind >>= 3

    return colormap.astype(np.uint8)


def draw_masks_to_frame(
    frame: np.ndarray, masks: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    masked_frame = frame
    for mask, color in zip(masks, colors):
        curr_masked_frame = np.where(mask[..., None], color, masked_frame)
        masked_frame = cv2.addWeighted(masked_frame, 0.75, curr_masked_frame, 0.25, 0)

        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )
        else:
            _, contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )

        cv2.drawContours(
            masked_frame, contours, -1, (255, 255, 255), 7
        )
        cv2.drawContours(
            masked_frame, contours, -1, (0, 0, 0), 5
        )
        cv2.drawContours(
            masked_frame, contours, -1, color.tolist(), 3
        )
    return masked_frame


def get_annot_df(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    dfs = {}

    for k, v in data.items():
        if k in ("info", "licenses"):
            dfs[k] = v
            continue
        df = pd.DataFrame(v)
        dfs[k] = df

    return dfs


def get_annot_dfs(file_list: list[str]):
    dfs = {}
    for annot_file in tqdm(file_list):
        dataset_name = Path(annot_file).stem
        dfs[dataset_name] = get_annot_df(annot_file)

    return dfs


def get_media_dir(media_dir: str, dataset: str):
    if dataset in ["saco_veval_sav_test", "saco_veval_sav_val"]:
        return os.path.join(media_dir, "saco_sav", "JPEGImages_24fps")
    elif dataset in ["saco_veval_yt1b_test", "saco_veval_yt1b_val"]:
        return os.path.join(media_dir, "saco_yt1b", "JPEGImages_6fps")
    elif dataset in ["saco_veval_smartglasses_test", "saco_veval_smartglasses_val"]:
        return os.path.join(media_dir, "saco_sg", "JPEGImages_6fps")
    elif dataset == "sa_fari_test":
        return os.path.join(media_dir, "sa_fari", "JPEGImages_6fps")
    else:
        raise ValueError(f"Dataset {dataset} not found")


def get_all_annotations_for_frame(
    dataset_df: pd.DataFrame, video_id: int, frame_idx: int, data_dir: str, dataset: str
):
    media_dir = os.path.join(data_dir, "media")

    annot_df = dataset_df["annotations"]
    video_df = dataset_df["videos"]

    video_df_current = video_df[video_df.id == video_id]
    assert (
        len(video_df_current) == 1
    ), f"Expected 1 video row, got {len(video_df_current)}"
    video_row = video_df_current.iloc[0]
    file_name = video_row.file_names[frame_idx]
    file_path = os.path.join(
        get_media_dir(media_dir=media_dir, dataset=dataset), file_name
    )
    frame = cv2.imread(file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    annot_df_current_video = annot_df[annot_df.video_id == video_id]
    if len(annot_df_current_video) == 0:
        print(f"No annotations found for video_id {video_id}")
        return frame, None, None
    else:
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_np_pairs = annot_df_current_video.apply(
            lambda row: (
                (
                    mask_utils.decode(row.segmentations[frame_idx])
                    if row.segmentations[frame_idx]
                    else empty_mask
                ),
                row.noun_phrase,
            ),
            axis=1,
        )
        mask_np_pairs = sorted(mask_np_pairs, key=lambda x: x[1])
        masks, noun_phrases = zip(*mask_np_pairs)

    return frame, masks, noun_phrases


def visualize_prompt_overlay(
    frame_idx,
    video_frames,
    title="Prompt Visualization",
    text_prompt=None,
    point_prompts=None,
    point_labels=None,
    bounding_boxes=None,
    box_labels=None,
    obj_id=None,
):
    """Simple prompt visualization function"""
    img = Image.fromarray(load_frame(video_frames[frame_idx]))
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.imshow(img)

    img_w, img_h = img.size

    if text_prompt:
        ax.text(
            0.02,
            0.98,
            f'Text: "{text_prompt}"',
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
            verticalalignment="top",
        )

    if point_prompts:
        for i, point in enumerate(point_prompts):
            x, y = point
            x_img, y_img = x * img_w, y * img_h

            if point_labels and len(point_labels) > i:
                color = "green" if point_labels[i] == 1 else "red"
                marker = "o" if point_labels[i] == 1 else "x"
            else:
                color = "green"
                marker = "o"

            ax.plot(
                x_img,
                y_img,
                marker=marker,
                color=color,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )
            ax.text(
                x_img + 5,
                y_img - 5,
                f"P{i+1}",
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    if bounding_boxes:
        for i, box in enumerate(bounding_boxes):
            x, y, w, h = box
            x_img, y_img = x * img_w, y * img_h
            w_img, h_img = w * img_w, h * img_h

            if box_labels and len(box_labels) > i:
                color = "green" if box_labels[i] == 1 else "red"
            else:
                color = "green"

            rect = patches.Rectangle(
                (x_img, y_img),
                w_img,
                h_img,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_img,
                y_img - 5,
                f"B{i+1}",
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    if obj_id is not None:
        ax.text(
            0.02,
            0.02,
            f"Object ID: {obj_id}",
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7),
            verticalalignment="bottom",
        )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _compute_rowcol_ids_xyxy(boxes_xyxy, fixed_cols=None, row_threshold_ratio=1):
    """
    根据边界框 XYXY 绝对坐标，生成 '行_列' 的编号（m_n）。
    - fixed_cols: 若每行固定列数，给一个正整数；否则自动分行
    - row_threshold_ratio: 自动分行阈值 = 0.6 * 中位数框高，可按需要调整
    返回: dict {原始下标 i: 'm_n'}
    """
    boxes = np.asarray(boxes_xyxy, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return {i: str(i + 1) for i in range(len(boxes))}
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h  = (y2 - y1)

    order = np.lexsort((cx, cy))  # 先按 y 再按 x：上到下、每行从左到右
    ids_map = {}

    if fixed_cols and fixed_cols > 0:
        for rank, orig_idx in enumerate(order):
            r = rank // fixed_cols + 1
            c = rank %  fixed_cols + 1
            ids_map[int(orig_idx)] = f"{r}_{c}"
        return ids_map

    # 自动分行：根据 y 中心的跳变（与框高相关）
    median_h = np.median(h) if len(h) else 1.0
    row_thr  = max(1.0, row_threshold_ratio * median_h)

    rows, current, last_cy = [], [], None
    for orig_idx in order:
        i = int(orig_idx)
        if last_cy is None:
            current = [i]; last_cy = cy[i]; continue
        if abs(cy[i] - last_cy) > row_thr:      # 认为到了下一行
            current.sort(key=lambda j: cx[j]); rows.append(current)
            current = [i]; last_cy = cy[i]
        else:
            current.append(i)
    if current:
        current.sort(key=lambda j: cx[j]); rows.append(current)

    for r, row_idxs in enumerate(rows, start=1):
        for c, i in enumerate(row_idxs, start=1):
            ids_map[int(i)] = f"{r}_{c}"
    return ids_map


def plot_results(img, results, save_name=None, use_rowcol_ids=True, fixed_cols=None):
    """
    可视化函数：仅负责绘制传入的 results。
    过滤逻辑（圆度、连通域等）应在调用此函数前处理完毕（例如在 postprocess 模块中）。
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")

    # 计算每个实例的 m_n（行_列）编号
    if use_rowcol_ids and nb_objects > 0:
        boxes_xyxy = results["boxes"].cpu().numpy()  # XYXY 绝对坐标
        ids_map = _compute_rowcol_ids_xyxy(boxes_xyxy, fixed_cols=fixed_cols)
    else:
        ids_map = {i: str(i + 1) for i in range(nb_objects)}  # 回退：1..N

    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]

        mask_t  = results["masks"][i].squeeze(0)
        mask_np = mask_t.detach().cpu().numpy()
        # [修改点] 直接绘制 mask_np，因为数据已在 main.py 的 postprocess 阶段清洗干净
        plot_mask(mask_np, color=color)

        w, h = img.size
        prob = results["scores"][i].item()
        nice_id = ids_map.get(i, str(i + 1))  # ← 行_列（m_n）

        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={nice_id}, prob={prob:.2f})",  # ← 显示 m_n
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )

    if save_name is not None:
        save_path = Path(save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

# ====== [保持不变] 形状/颜色/纹理特征与 CSV 导出 ======
def _mask_to_uint8_np(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    t = np.array(t)
    if t.ndim == 3:
        t = t.squeeze(0)
    return (t > 0.5).astype(np.uint8)

def _compute_shape_features(mask: np.ndarray):
    m = mask.astype(np.uint8)
    canopy_area = float(m.sum())

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return dict(
            canopy_area=0.0,
            canopy_perimeter=0.0,
            circularity=0.0,
            convex_hull_area=0.0,
            min_rect_long_axis=0.0,
        )
    cnt = max(contours, key=cv2.contourArea)
    canopy_perimeter = float(cv2.arcLength(cnt, True))
    circularity = float(4.0 * math.pi * canopy_area / (canopy_perimeter ** 2)) if canopy_perimeter > 1e-6 else 0.0

    hull = cv2.convexHull(cnt)
    convex_hull_area = float(cv2.contourArea(hull))

    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    min_rect_long_axis = float(max(w, h))

    return dict(
        canopy_area=canopy_area,
        canopy_perimeter=canopy_perimeter,
        circularity=circularity,
        convex_hull_area=convex_hull_area,
        min_rect_long_axis=min_rect_long_axis,
    )

def _compute_color_indices(img_rgb: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool)
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    eps = 1e-6

    ngbdi = (G - B) / (G + B + eps)
    exg   = 2 * G - R - B
    exr   = 1.4 * R - G
    exg_exr_diff = exg - exr
    tgi   = G - 0.39 * R - 0.61 * B

    def _mean_in_mask(x):
        vals = x[m]
        return float(vals.mean()) if vals.size else 0.0

    return dict(
        ngbdi=_mean_in_mask(ngbdi),
        exg_exr_diff=_mean_in_mask(exg_exr_diff),
        tgi=_mean_in_mask(tgi),
        exg=_mean_in_mask(exg),
        exr=_mean_in_mask(exr),
    )

def _compute_texture_features(img_rgb: np.ndarray, mask: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vals = gray[mask.astype(bool)]
    if vals.size == 0:
        return dict(
            third_moment=0.0,
            low_gray_level_emphasis=0.0,
            high_gray_level_emphasis=0.0,
            gray_entropy=0.0,
            gray_non_uniformity=0.0,
        )
    vals = vals.astype(np.float32)
    mu = vals.mean()
    third_moment = float(np.mean((vals - mu) ** 3))

    hist, _ = np.histogram(vals, bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    s = p.sum()
    if s > 0:
        p /= s
    i = np.arange(256, dtype=np.float64)
    low_gray_level_emphasis  = float(np.sum(p / ((i + 1.0) ** 2)))
    high_gray_level_emphasis = float(np.sum(p * ((i + 1.0) ** 2)))
    nz = p[p > 0]
    gray_entropy = float(-(nz * np.log(nz)).sum())
    gray_non_uniformity = float(np.sum(p ** 2))

    return dict(
        third_moment=third_moment,
        low_gray_level_emphasis=low_gray_level_emphasis,
        high_gray_level_emphasis=high_gray_level_emphasis,
        gray_entropy=gray_entropy,
        gray_non_uniformity=gray_non_uniformity,
    )

def extract_features_and_save_csv(img_pil, results, csv_path, fixed_cols=None):
    if isinstance(img_pil, Image.Image):
        img_np = np.array(img_pil.convert("RGB"))
    else:
        img_np = np.array(img_pil, dtype=np.uint8)
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    N = len(results["scores"])

    english_cols = [
        "plant_id",
        "canopy_area","canopy_perimeter","circularity",
        "convex_hull_area","min_rect_long_axis",
        "ngbdi","exg_exr_diff","tgi","exg","exr",
        "third_moment","low_gray_level_emphasis","high_gray_level_emphasis",
        "gray_entropy","gray_non_uniformity",
    ]
    chinese_cols = [
        "植株编号",
        "冠层投影面积","冠层投影周长","圆形度",
        "冠层凸包面积","外接矩形长轴长度",
        "归一化绿蓝差异指数","超绿超红差分指数","三角绿度指数","超绿指数","超红指数",
        "三阶矩","低灰度强调","高灰度强调","灰度熵","灰度非均匀性",
    ]

    if N == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(english_cols) + "\n")
            f.write(",".join(chinese_cols) + "\n")
        return

    boxes_xyxy = results["boxes"].cpu().numpy()
    ids_map = _compute_rowcol_ids_xyxy(boxes_xyxy, fixed_cols=fixed_cols)

    rows = []
    for i in range(N):
        mask = _mask_to_uint8_np(results["masks"][i])
        shape_dic   = _compute_shape_features(mask)
        color_dic   = _compute_color_indices(img_np, mask)
        texture_dic = _compute_texture_features(img_np, mask)

        row = {
            "plant_id": ids_map.get(i, str(i + 1)),
            **shape_dic,
            **color_dic,
            **texture_dic,
        }
        rows.append(row)

    df = pd.DataFrame(rows)[english_cols]
    csv_path = str(csv_path)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(english_cols) + "\n")
        f.write(",".join(chinese_cols) + "\n")
    df.to_csv(csv_path, index=False, header=False, mode="a", encoding="utf-8")
    print(f"[FEATURE] CSV saved to: {csv_path}")


def single_visualization(img, anns, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    overlay = render_masklet_frame(img, anns, alpha=0.5)
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout()


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def load_frame(frame):
    if isinstance(frame, np.ndarray):
        img = frame
    elif isinstance(frame, Image.Image):
        img = np.array(frame)
    elif isinstance(frame, str) and os.path.isfile(frame):
        img = plt.imread(frame)
    else:
        raise ValueError(f"Invalid video frame type: {type(frame)=}")
    return img