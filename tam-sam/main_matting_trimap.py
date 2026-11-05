
# main_trimap_T_SAM3.py
# 功能：
# 1. SAM3输出：每个实例mask分别生成trimap
# 2. T-SAM3输出：调用trimapsegment.py中的TrimapSegmenter，
#    对每个实例mask分别生成真实T-SAM3 trimap
#
# 输出：
# SAM3:
#   position1_SAM3_trimap_1.png ...
#
# T-SAM3:
#   position1_T_SAM3_trimap_1.png ...

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from sam3.trimapsegment import (
    TrimapSegmenter,
    TrimapSegmentConfig
)


CHECKPOINT_PATH = "/root/autodl-tmp/sam3.pt"

IMAGE_PATH = Path(
    "/root/sam3-main/assets_p1/6.4/zao/position1.png"
)

OUTPUT_DIR = Path(
    "/root/autodl-tmp/plot_results/results_matting/trimap/6.4/zao"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PROMPT = "pine seedling leaves, needle leaves"


# ==============================
# SAM3基础trimap参数
# ==============================

TRIMAP_KERNEL_SIZE = 15
ERODE_ITERATIONS = 1
DILATE_ITERATIONS = 1


def binary_mask_to_trimap(
    mask,
    kernel_size=15,
    erode_iterations=1,
    dilate_iterations=1
):

    import cv2

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    sure_fg = cv2.erode(
        mask,
        kernel,
        iterations=erode_iterations
    )

    dilated = cv2.dilate(
        mask,
        kernel,
        iterations=dilate_iterations
    )

    # 背景255，unknown128，前景0
    trimap = np.full_like(
        mask,
        255,
        dtype=np.uint8
    )

    trimap[dilated > 0] = 128
    trimap[sure_fg > 0] = 0

    return trimap



def save_instance_trimap(
    masks,
    save_prefix,
    func
):

    count = 0

    for i in range(
        masks.shape[0]
    ):

        mask_i = masks[i] > 0

        if not np.any(mask_i):
            continue

        count += 1

        trimap = func(mask_i)

        save_path = (
            OUTPUT_DIR /
            f"{save_prefix}_{count}.png"
        )

        Image.fromarray(
            trimap.astype(np.uint8),
            mode="L"
        ).save(
            save_path
        )

        print(
            "[OK]",
            save_path
        )

    print(
        f"{save_prefix} count:",
        count
    )



def main():

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")


    print("[INFO] Loading SAM3")

    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH
    )


    processor = Sam3Processor(
        model,
        confidence_threshold=0.45
    )


    state = processor.set_image(
        image
    )


    output = processor.set_text_prompt(
        state=state,
        prompt=PROMPT
    )


    masks = output["masks"]

    masks = (
        masks.detach()
        .cpu()
        .numpy()
    )


    # [N,1,H,W] -> [N,H,W]
    if masks.ndim == 4:
        masks = masks[:,0]


    print(
        "[INFO] SAM3 instances:",
        masks.shape[0]
    )


    # --------------------------------
    # 1. SAM3 mask trimap
    # --------------------------------

    save_instance_trimap(
        masks,
        "position1_SAM3_trimap",
        lambda x:
            binary_mask_to_trimap(
                x,
                TRIMAP_KERNEL_SIZE,
                ERODE_ITERATIONS,
                DILATE_ITERATIONS
            )
    )


    # --------------------------------
    # 2. T-SAM3真实trimap
    # --------------------------------

    cfg = TrimapSegmentConfig(
        erode_iters=5,
        dilate_iters=5,
        kernel_size=3,
        alpha_threshold=0.5,
        fba_weights=
        "/root/sam3-main/"
        "sam3/third_party/"
        "FBA_Matting/FBA.pth",
        fba_repo_dir=
        "/root/sam3-main/"
        "sam3/third_party/"
        "FBA_Matting",
        fba_scale=1.0
    )


    segmenter = TrimapSegmenter(
        config=cfg,
        device="cuda"
    )


    save_instance_trimap(
        masks,
        "position1_T_SAM3_trimap",
        lambda x:
            segmenter._build_trimap_3(x)
    )


    print("[DONE]")


if __name__ == "__main__":
    main()
